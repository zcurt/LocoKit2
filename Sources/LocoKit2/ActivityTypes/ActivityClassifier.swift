//
//  ActivityClassifier.swift
//  
//
//  Created by Matt Greenfield on 2/9/22.
//

import Foundation
import CoreLocation
import CoreML
import UIKit
import Surge
import GRDB

@ActivityTypesActor
public enum ActivityClassifier {
    
    // MARK: - Classifying

    public static func canClassify(_ coordinate: CLLocationCoordinate2D? = nil) -> Bool {
        if let coordinate {
            refreshModels(for: coordinate)
        }
        return !models.isEmpty
    }

    public static func results(for sample: LocomotionSample) async -> ClassifierResults? {
        if let cached = cache.object(forKey: sample.id as NSString) {
            return cached
        }

        // no Core ML in background plz
        if await UIApplication.shared.applicationState == .background { return nil }

        // make sure have suitable classifiers
        if let coordinate = sample.location?.coordinate {
            refreshModels(for: coordinate)
        }

        // highest priorty first (ie CD2 first)
        let classifiers = models.sorted { $0.key > $1.key }.map { $0.value }

        // weighted mean: each model claims its completeness share of the weight still
        // unclaimed, and the last one (normally the base model) takes the rest — so a
        // thin regional model no longer speaks for everyone just by being asked first
        var scores: [ActivityType: Double] = [:]
        var totalWeight = 0.0
        var remainingWeight = 1.0

        for classifier in classifiers {
            let results = await classifier.classify(sample)
            if results.resultItems.isEmpty { continue } // no model file yet: no opinion, no weight

            let isLast = classifier.id == classifiers.last?.id
            let share = if isLast {
                classifier.geoKey.hasPrefix("B") ? baseModelWeight : 1.0
            } else {
                classifier.completenessScore
            }
            let weight = remainingWeight * share
            if weight <= 0 { continue }

            for item in results.resultItems {
                scores[item.activityType, default: 0] += item.score * weight
            }
            totalWeight += weight
            remainingWeight -= weight

            if remainingWeight <= 0 { break }
        }

        let combinedResults: ClassifierResults? = totalWeight > 0
            ? ClassifierResults(resultItems: scores.map { ClassifierResultItem(name: $0.key, score: $0.value / totalWeight) })
            : nil

        if let combinedResults {
            cache.setObject(combinedResults, forKey: sample.id as NSString)
        }

        return combinedResults
    }

    public static func results(for samples: [LocomotionSample], timeout: TimeInterval? = nil) async -> (combinedResults: ClassifierResults?, perSampleResults: [String: ClassifierResults])? {
        if samples.isEmpty { return nil }

        // no Core ML in background plz
        if await UIApplication.shared.applicationState == .background { return nil }
        
        guard let handle = OperationRegistry.startOperation(
            .activityTypes,
            operation: "ActivityClassifier.results(for:timeout:)",
            objectKey: samples.hashValue.description
        ) else { return nil }

        defer { OperationRegistry.endOperation(handle) }

        let start = Date()

        var allScores: [ActivityType: [Double]] = [:]
        for typeName in ActivityType.allCases {
            allScores[typeName] = []
        }

        var perSampleResults: [String: ClassifierResults] = [:]

        for sample in samples {
            if let timeout, start.age >= timeout {
                Log.info("ActivityClassifier reached timeout limit (\(timeout) seconds)", subsystem: .activitytypes)
                return nil  // abort with nil on timeout - no partial results
            }

            guard let results = await results(for: sample) else {
                continue
            }

            perSampleResults[sample.id] = results

            for typeName in ActivityType.allCases {
                if let resultRow = results[typeName] {
                    allScores[resultRow.activityType]!.append(resultRow.score)
                } else {
                    allScores[typeName]!.append(0)
                }
            }
        }

        var finalResults: [ClassifierResultItem] = []

        for typeName in ActivityType.allCases {
            var finalScore = 0.0
            if let scores = allScores[typeName], !scores.isEmpty {
                finalScore = mean(scores)
            }

            finalResults.append(ClassifierResultItem(name: typeName, score: finalScore))
        }

        return (ClassifierResults(resultItems: finalResults), perSampleResults)
    }

    // MARK: - Base model

    /// Share of the remaining weight the base model (BD0) takes when it's the last
    /// classifier consulted. 0.5 suits a placeholder; an app that ships a base model
    /// trained on real history can raise it to 1.
    nonisolated(unsafe) public static var baseModelWeight: Double = 0.5

    /// Drops cached results and models — call after the base model file changes.
    public static func baseModelChanged() {
        if let base = models.first(where: { $0.value.geoKey.hasPrefix("B") })?.value {
            MLModelCache.invalidateModelFor(filename: base.filename)
        }
        models = models.filter { !$0.value.geoKey.hasPrefix("B") }
        cache.removeAllObjects()
    }

    // MARK: - Results caching

    private static let cache = NSCache<NSString, ClassifierResults>()

    private static func set(results: ClassifierResults, sampleId: String) {
        cache.setObject(results, forKey: sampleId as NSString)
    }

    // MARK: - Fetching models

    public private(set) static var models: [Int: ActivityTypesModel] = [:] // index = priority

    private static func refreshModels(for coordinate: CLLocationCoordinate2D) {
        var updated = models.filter { (key, classifier) in
            return classifier.contains(coordinate: coordinate)
        }

        let baseModelURL = MLModelCache.baseModelURL()
        let targetModelsCount = baseModelURL != nil ? 4 : 3

        // all existing classifiers are good?
        if updated.count == targetModelsCount { return }

        // get a CD2
        if updated.first(where: { $0.value.geoKey.hasPrefix("CD2") == true }) == nil {
            updated[2] = ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 2) // priority 2 (top)
        }

        // get a CD1
        if updated.first(where: { $0.value.geoKey.hasPrefix("CD1") == true }) == nil {
            updated[1] = ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 1)
        }
        
        // get a CD0
        if updated.first(where: { $0.value.geoKey.hasPrefix("CD0") == true }) == nil {
            updated[0] = ActivityTypesModel.fetchModelFor(coordinate: coordinate, depth: 0)
        }

        // get the base model (BD0): the app's retrained copy, else the bundled one
        if let baseModelURL, updated.first(where: { $0.value.geoKey.hasPrefix("BD0") == true }) == nil {
            updated[-1] = ActivityTypesModel(bundledURL: baseModelURL)
        }

        models = updated
    }

    public static func invalidateModel(geoKey: String) {
        if let model = models.first(where: { $0.value.geoKey == geoKey })?.value {
            MLModelCache.invalidateModelFor(filename: model.filename)
        }
        models = models.filter { $0.value.geoKey != geoKey }
    }
    
    public static func clearModels() {
        models = [:]
        cache.removeAllObjects()
    }

}
