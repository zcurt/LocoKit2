//
//  ActivityTypesManager.swift
//  
//
//  Created by Matt Greenfield on 5/11/22.
//

import Foundation
import BackgroundTasks
import CoreLocation
import TabularData
import CoreML
import GRDB
#if !targetEnvironment(simulator)
import CreateML
#endif

@ActivityTypesActor
public enum ActivityTypesManager {
    
    // MARK: - Task Configuration

    nonisolated
    public static let taskIdentifier = "com.bigpaua.Arc.activityTypeModelUpdates"

    // MARK: - Queueing Model Updates

    public static func queueUpdatesForModelsContaining(_ samples: [LocomotionSample]) {
        var lastD2Model: ActivityTypesModel?
        var models: Set<ActivityTypesModel> = []

        for sample in samples where sample.confirmedActivityType != nil {
            guard sample.hasUsableCoordinate, let coordinate = sample.location?.coordinate else { continue }

            if let lastD2Model, lastD2Model.contains(coordinate: coordinate) {
                continue
            }

            let d2model = ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 2)
            models.insert(d2model)
            lastD2Model = d2model

            models.insert(ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 1))
            models.insert(ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 0))
        }

        do {
            try Database.pool.write { db in
                for var model in models {
                    try model.updateChanges(db) {
                        $0.needsUpdate = true
                    }
                }
            }

        } catch {
            Log.error(error, subsystem: .database)
        }
    }

    // MARK: - Background Task Management

    @MainActor
    public static func registerModelUpdatesTask() {
        let taskDefinition = BackgroundTaskDefinition(
            identifier: taskIdentifier,
            displayName: "activity model updates",
            minimumDelay: .hours(1),
            requiresNetwork: false,
            requiresPower: true,
            foregroundThreshold: .days(2),
            workHandler: processModelsForBackground
        )
        
        BackgroundTasksManager.add(task: taskDefinition)
    }
    
    private static func processModelsForBackground() async throws {
        while true {
            if Task.isCancelled { throw CancellationError() }
            
            // get prioritised model to update (one at a time)
            let model = await fetchNextModelToUpdate()
            
            // no more models to update? we're done
            guard let model else { return }
            
            await updateModel(geoKey: model.geoKey)
        }
    }
    
    private static func fetchNextModelToUpdate() async -> ActivityTypesModel? {

        // CD0 update interval for already "complete" models
        let cd0UpdateInterval: TimeInterval = .days(7)

        // CD0 update interval for "incomplete" moels
        let cd0FrequentUpdateInterval: TimeInterval = .days(1)

        do {
            return try await Database.pool.read { db in
                try ActivityTypesModel
                    .filter(
                        sql: """
                        needsUpdate = 1 AND 
                        (depth > 0 OR 
                         (depth = 0 AND 
                          (lastUpdated IS NULL OR 
                           (totalSamples < ? AND lastUpdated < datetime('now', '-\(Int(cd0FrequentUpdateInterval)) seconds')) OR
                           (totalSamples >= ? AND lastUpdated < datetime('now', '-\(Int(cd0UpdateInterval)) seconds'))
                          )
                         )
                        )
                        """,
                        arguments: [ActivityTypesModel.modelMinTrainingSamples[0]!, ActivityTypesModel.modelMinTrainingSamples[0]!]
                    )
                    .order { [$0.depth.desc, $0.totalSamples.asc] }
                    .fetchOne(db)
            }
            
        } catch {
            Log.error(error, subsystem: .database)
            return nil
        }
    }
    
    public static func fetchPendingModelGeoKeys() async throws -> [String] {
        return try await Database.pool.read { db in
            let request = ActivityTypesModel
                .select(\.geoKey)
                .filter { $0.needsUpdate == true }
                .order { [$0.depth.desc, $0.totalSamples.asc] }
            return try String.fetchAll(db, request)
        }
    }
    
    public static func deleteAllModels() async throws {
        // first clear from memory cache
        ActivityClassifier.clearModels()
        
        // then remove from database
        try await Database.pool.write { db in
            _ = try ActivityTypesModel.deleteAll(db)
        }
        
        // then delete model files
        let manager = FileManager.default
        if let files = try? manager.contentsOfDirectory(at: MLModelCache.modelsDir, includingPropertiesForKeys: nil) {
            for file in files {
                if file.lastPathComponent.hasPrefix("CD") {
                    try? manager.removeItem(at: file)
                }
            }
        }
        
        Log.info("Deleted all ActivityTypesModels", subsystem: .activitytypes)
    }

    public static func updateModel(geoKey: String) async {
        do {
            let model = try await Database.pool.read {
                try ActivityTypesModel.fetchOne($0, key: geoKey)
            }
            if let model {
                guard let handle = OperationRegistry.startOperation(
                    .activityTypes,
                    operation: "ActivityTypesManager.updateModel(geoKey:)",
                    objectKey: model.geoKey,
                    rejectDuplicates: true,
                    maxConcurrent: maxConcurrentModelUpdates
                ) else {
                    Log.debug("Skipping duplicate ActivityTypesManager.updateModel(geoKey:) for \(model.geoKey)", subsystem: .activitytypes)
                    return
                }
                defer { OperationRegistry.endOperation(handle) }

                // remove from queue immediately, before the detached training task begins,
                // to prevent the background loop from re-fetching this model while it's being updated
                try await Database.pool.write { db in
                    var mutableModel = model
                    try mutableModel.updateChanges(db) {
                        $0.needsUpdate = false
                    }
                }

                await update(model: model)
            }
            
        } catch {
            Log.error(error, subsystem: .database)
        }
    }
    
    static let maxConcurrentModelUpdates = 3

    public static func processModelUpdate(model: ActivityTypesModel, fileMissing: Bool = false) {
        guard model.needsUpdate else { return }

        let shouldUpdateImmediately = fileMissing || (model.depth == 2 && model.completenessScore < 0.1)

        if shouldUpdateImmediately {
            guard OperationRegistry.highlander.operationCount(for: .activityTypes) < maxConcurrentModelUpdates else { return }
            let geoKey = model.geoKey
            Task(priority: .utility) { await updateModel(geoKey: geoKey) }
        }
    }

    // MARK: - Model building

#if targetEnvironment(simulator)
    static func update(model: ActivityTypesModel) async {
        print("SIMULATOR DOESN'T SUPPORT MODEL UPDATES")
    }
#else
    static func update(model: ActivityTypesModel) async {
        if model.geoKey.hasPrefix("B") { return }
        if Task.isCancelled { return }

        // run heavy training work off-actor to avoid blocking classification
        let trained = await Task.detached(priority: .utility) { [model] () -> Bool in
            Log.debug("UPDATING: \(model.geoKey)", subsystem: .activitytypes)

            let manager = FileManager.default
            let tempModelFile = manager.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).mlmodel")
            var csvFile: URL?

            // BIG-530: clean up training artefacts in tmp/ on all exit paths
            // (success, throw, early return). iOS doesn't actively purge tmp/,
            // so files left here accumulate indefinitely — single training cycle
            // can leave ~40-50MB CSV behind, hundreds of those become GBs.
            defer {
                try? manager.removeItem(at: tempModelFile)
                if let csvFile { try? manager.removeItem(at: csvFile) }
            }

            do {
                var samplesCount = 0
                var includedTypes: Set<ActivityType> = []

                let start = Date()
                let samples = try await fetchTrainingSamples(for: model)
                Log.debug("UPDATING: \(model.geoKey), SAMPLES BATCH: \(samples.count), duration: \(start.age)", subsystem: .activitytypes)

                let (url, samplesAdded, typesAdded) = try exportCSV(samples: samples, appendingTo: csvFile)
                Log.debug("UPDATING: \(model.geoKey), CSV EXPORT: \(samplesAdded) samples, elapsed: \(start.age)", subsystem: .activitytypes)
                csvFile = url
                samplesCount += samplesAdded
                includedTypes.formUnion(typesAdded)

                // if includedTypes only has one type and it's not stationary, throw in a fake stationary sample
                if samplesCount > 0 && includedTypes.count == 1 && !includedTypes.contains(.stationary) {
                    Log.debug("UPDATING: \(model.geoKey), ADDING FAKE STATIONARY SAMPLE", subsystem: .activitytypes)

                    let fakeLocation = CLLocation(
                        coordinate: model.centerCoordinate,
                        altitude: 0,
                        horizontalAccuracy: 10,
                        verticalAccuracy: 10,
                        course: 0,
                        speed: 0,
                        timestamp: .now
                    )

                    var fakeSample = LocomotionSample(
                        date: .now,
                        movingState: .stationary,
                        recordingState: .recording,
                        location: fakeLocation
                    )
                    fakeSample.confirmedActivityType = .stationary
                    fakeSample.xyAcceleration = 0
                    fakeSample.zAcceleration = 0
                    fakeSample.stepHz = 0

                    let (url, samplesAdded, typesAdded) = try exportCSV(samples: [fakeSample], appendingTo: csvFile)
                    csvFile = url
                    samplesCount += samplesAdded
                    includedTypes.formUnion(typesAdded)
                }

                guard samplesCount > 0, includedTypes.count > 1 else {
                    Log.debug("SKIPPED: \(model.geoKey) (samples: \(samplesCount), includedTypes: \(includedTypes.count))", subsystem: .activitytypes)
                    try? await Database.pool.write { [samplesCount] db in
                        var mutableModel = model
                        try mutableModel.updateChanges(db) {
                            $0.totalSamples = samplesCount
                            $0.accuracyScore = nil
                            $0.lastUpdated = .now
                            $0.needsUpdate = false
                        }
                    }
                    return false
                }

                guard let csvFile else {
                    Log.error("Missing CSV file for model build", subsystem: .activitytypes)
                    return false
                }

                let dataFrame = try DataFrame(contentsOfCSVFile: csvFile)
                Log.debug("UPDATING: \(model.geoKey), TRAINING START: \(samplesCount) samples, \(includedTypes.count) types, elapsed: \(start.age)", subsystem: .activitytypes)
                let classifier = try MLBoostedTreeClassifier(trainingData: dataFrame, targetColumn: "confirmedActivityType")
                Log.debug("UPDATING: \(model.geoKey), TRAINING DONE, elapsed: \(start.age)", subsystem: .activitytypes)

                do {
                    try FileManager.default.createDirectory(at: MLModelCache.modelsDir, withIntermediateDirectories: true, attributes: nil)
                } catch {
                    Log.error("Couldn't create MLModels directory", subsystem: .activitytypes)
                }

                try classifier.write(to: tempModelFile)
                let compiledModelFile = try await MLModel.compileModel(at: tempModelFile)
                _ = try manager.replaceItemAt(MLModelCache.getModelURLFor(filename: model.filename), withItemAt: compiledModelFile)
                Log.debug("UPDATING: \(model.geoKey), COMPILE DONE, elapsed: \(start.age)", subsystem: .activitytypes)

                let accuracy = 1.0 - classifier.validationMetrics.classificationError
                try? await Database.pool.write { [samplesCount] db in
                    var mutableModel = model
                    try mutableModel.updateChanges(db) {
                        $0.totalSamples = samplesCount
                        $0.accuracyScore = accuracy
                        $0.lastUpdated = .now
                        $0.needsUpdate = false
                    }
                }

                let completeness = min(1.0, Double(samplesCount) / Double(ActivityTypesModel.modelMinTrainingSamples[model.depth]!))
                Log.info("UPDATED: \(model.geoKey) (samples: \(samplesCount), accuracy: \(String(format: "%.2f", accuracy)), completeness: \(String(format: "%.2f", completeness)), includedTypes: \(includedTypes.count))", subsystem: .activitytypes)

                return true

            } catch {
                Log.error(error, subsystem: .activitytypes)
                return false
            }
        }.value

        if Task.isCancelled { return }

        // only cache reload and classifier invalidation need the actor
        if trained {
            try? model.reloadModel()
            ActivityClassifier.invalidateModel(geoKey: model.geoKey)
        }
    }
#endif

    // MARK: - Base model (BD0) training

    nonisolated
    public static let baseModelTaskIdentifier = "com.bigpaua.Arc.baseModelUpdates"

    /// Registers a weekly (on power) retrain of the base model from `seedCSV` plus this
    /// device's confirmed samples. The app must list the identifier in Info.plist's
    /// BGTaskSchedulerPermittedIdentifiers.
    @MainActor
    public static func registerBaseModelTask(seedCSV: URL?) {
        BackgroundTasksManager.add(task: BackgroundTaskDefinition(
            identifier: baseModelTaskIdentifier,
            displayName: "base activity model update",
            minimumDelay: .days(7),
            requiresNetwork: false,
            requiresPower: true,
            foregroundThreshold: .days(14),
            workHandler: { _ = try await trainBaseModel(seedCSV: seedCSV) }
        ))
    }

#if targetEnvironment(simulator)
    public nonisolated static func trainBaseModel(seedCSV: URL? = nil) async throws -> URL {
        throw NSError(domain: "ActivityTypes", code: 0, userInfo: [NSLocalizedDescriptionKey: "Base model training requires a real device"])
    }
#else
    /// Trains the location-free base model (BD0) from `seedCSV` (rows in
    /// `ActivityTypesModel.baseModelCSVHeader` form — e.g. bundled history from another
    /// recorder) plus every confirmed sample on this device, folded into the seven
    /// `bd0Bucket`s. The result replaces MLModels/BD0.mlmodelc, which the classifier
    /// prefers over a bundled copy.
    public nonisolated static func trainBaseModel(seedCSV: URL? = nil) async throws -> URL {
        try await Task.detached(priority: .utility) {
            // BIG-530: clean up training artefacts in tmp/ on all exit paths
            let manager = FileManager.default
            let csvFile = manager.temporaryDirectory.appendingPathComponent("BD0_training.csv")
            let tempModelFile = manager.temporaryDirectory.appendingPathComponent("BD0.mlmodel")
            var compiledModelFile: URL?
            defer {
                try? manager.removeItem(at: csvFile)
                try? manager.removeItem(at: tempModelFile)
                if let compiledModelFile { try? manager.removeItem(at: compiledModelFile) }
            }

            var csv: String
            if let seedCSV {
                csv = try String(contentsOf: seedCSV, encoding: .utf8)
                if !csv.hasSuffix("\n") { csv += "\n" }
            } else {
                csv = ActivityTypesModel.baseModelCSVHeader + "\n"
            }

            let samples = try await Database.pool.read { db in
                try LocomotionSample
                    .filter(sql: """
                        confirmedActivityType IS NOT NULL
                        AND likely(xyAcceleration IS NOT NULL)
                        AND likely(zAcceleration IS NOT NULL)
                        AND likely(stepHz IS NOT NULL)
                        """)
                    .order(\.date.desc)
                    .limit(ActivityTypesModel.modelMaxTrainingSamples[0]!)
                    .fetchAll(db)
            }

            var localCount = 0
            for sample in samples {
                guard let row = ActivityTypesModel.baseModelCSVRow(for: sample) else { continue }
                csv += row + "\n"
                localCount += 1
            }
            try csv.write(to: csvFile, atomically: true, encoding: .utf8)

            let dataFrame = try DataFrame(contentsOfCSVFile: csvFile)
            let types = Set(dataFrame["confirmedActivityType"].compactMap { $0 as? Int })
            guard dataFrame.rows.count > 0, types.count > 1 else {
                throw NSError(domain: "ActivityTypes", code: 0, userInfo: [
                    NSLocalizedDescriptionKey: "Insufficient training data: \(dataFrame.rows.count) samples, \(types.count) types"
                ])
            }

            let classifier = try MLBoostedTreeClassifier(trainingData: dataFrame, targetColumn: "confirmedActivityType")
            let accuracy = 1.0 - classifier.validationMetrics.classificationError

            try? manager.removeItem(at: tempModelFile)
            try classifier.write(to: tempModelFile)
            compiledModelFile = try await MLModel.compileModel(at: tempModelFile)
            guard let compiledModelFile else { fatalError("unreachable: compiledModelFile assigned above") }

            try manager.createDirectory(at: MLModelCache.modelsDir, withIntermediateDirectories: true)
            let outputURL = MLModelCache.modelsDir.appendingPathComponent("BD0.mlmodelc")
            if manager.fileExists(atPath: outputURL.path) {
                _ = try manager.replaceItemAt(outputURL, withItemAt: compiledModelFile)
            } else {
                try manager.moveItem(at: compiledModelFile, to: outputURL)
            }

            await ActivityClassifier.baseModelChanged()

            Log.info("UPDATED: BD0 (rows: \(dataFrame.rows.count), local: \(localCount), accuracy: \(String(format: "%.3f", accuracy)))", subsystem: .activitytypes)
            return outputURL
        }.value
    }
#endif

    nonisolated
    private static func fetchTrainingSamples(for model: ActivityTypesModel) async throws -> [LocomotionSample] {
        return try await Database.pool.read { db in
            if model.depth != 0 {
                // Use spatial query with rtree subquery and forced index usage
                // This ensures the rtreeId index is used instead of the date index
                let sql = """
                    SELECT s.* FROM LocomotionSample s INDEXED BY LocomotionSample_on_rtreeId
                    WHERE s.rtreeId IN (
                        SELECT rowid FROM SampleRTree 
                        WHERE latMin >= ? AND latMax <= ? 
                        AND lonMin >= ? AND lonMax <= ?
                    )
                    AND s.confirmedActivityType IS NOT NULL
                    AND likely(s.xyAcceleration IS NOT NULL)
                    AND likely(s.zAcceleration IS NOT NULL)
                    AND likely(s.stepHz IS NOT NULL)
                    ORDER BY s.date DESC
                    LIMIT ?
                    """
                
                return try LocomotionSample.fetchAll(db, sql: sql, arguments: [
                    model.latitudeRange.lowerBound, model.latitudeRange.upperBound,
                    model.longitudeRange.lowerBound, model.longitudeRange.upperBound,
                    ActivityTypesModel.modelMaxTrainingSamples[model.depth]!
                ])

            } else {
                // For depth 0 (global), use the original approach
                let query = LocomotionSample
                    .filter(sql: """
                        confirmedActivityType IS NOT NULL
                        AND likely(xyAcceleration IS NOT NULL)
                        AND likely(zAcceleration IS NOT NULL)
                        AND likely(stepHz IS NOT NULL)
                        """)
                    .order(\.date.desc)
                    .limit(ActivityTypesModel.modelMaxTrainingSamples[model.depth]!)
                
                return try query.fetchAll(db)
            }
        }
    }

    nonisolated
    private static func exportCSV(samples: [LocomotionSample], appendingTo: URL? = nil) throws -> (URL, Int, Set<ActivityType>) {
        let modelFeatures = [
            "confirmedActivityType", "stepHz", "xyAcceleration", "zAcceleration", "movingState",
            "verticalAccuracy", "horizontalAccuracy", "speed", "course",
            "latitude", "longitude", "altitude", "heartRate",
            "timeOfDay", "sinceVisitStart"
        ]

        let csvFile = appendingTo ?? FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        // header the csv file
        if appendingTo == nil {
            try modelFeatures.joined(separator: ",").appendLineTo(csvFile)
        }

        var samplesAdded = 0
        var includedTypes: Set<ActivityType> = []

        // write the samples to file
        for sample in samples {
            guard let confirmedActivityType = sample.confirmedActivityType else { continue }
            guard let location = sample.location, location.hasUsableCoordinate else { continue }
            guard location.speed >= 0, location.course >= 0 else { continue }
            guard let stepHz = sample.stepHz else { continue }
            guard let xyAcceleration = sample.xyAcceleration else { continue }
            guard let zAcceleration = sample.zAcceleration else { continue }
            guard location.speed >= 0 else { continue }
            guard location.course >= 0 else { continue }
            guard location.horizontalAccuracy > 0 else { continue }
            guard location.verticalAccuracy > 0 else { continue }

            includedTypes.insert(sample.confirmedActivityType!)

            var line = ""
            line += "\(confirmedActivityType.rawValue),\(stepHz),\(xyAcceleration),\(zAcceleration),\(sample.movingState.rawValue),"
            line += "\(location.verticalAccuracy),\(location.horizontalAccuracy),\(location.speed),\(location.course),"
            line += "\(location.coordinate.latitude),\(location.coordinate.longitude),\(location.altitude),\(sample.heartRate ?? -1),"
            line += "\(sample.timeOfDay),\(sample.sinceVisitStart)"

            try line.appendLineTo(csvFile)
            samplesAdded += 1
        }

        return (csvFile, samplesAdded, includedTypes)
    }

}
