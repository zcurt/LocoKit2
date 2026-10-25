//
//  LocomotionSample+StrayFixes.swift
//  LocoKit2
//
//  Stray fixes: the few samples kilometres from the rest of an item — an
//  airport stop holding the first in-flight fixes, a GPS jump at home.
//  Left in, one of them pulls a visit's centre and stretches a place's
//  radius to its cap (Follow, 2026-10-25: an airport at 2 km, a lake
//  winning stops at a house). Taken out before geometry is computed.
//

import Foundation
import CoreLocation

extension Array where Element == LocomotionSample {

    /// The samples without stray fixes: those farther than `metres` from
    /// the per-axis median of the usable coordinates, when they are at most
    /// `maxShare` of the usable samples. Beyond that share the array comes
    /// back whole — the item holds movement, which is the processor's
    /// business, not geometry's.
    public func withoutStrayFixes(beyond metres: CLLocationDistance = 1000, maxShare: Double = 0.05) -> [LocomotionSample] {
        let usable = filter { $0.location?.coordinate.isUsable == true }
        guard usable.count >= 20 else { return self }
        let latitudes = usable.map { $0.location!.coordinate.latitude }.sorted()
        let longitudes = usable.map { $0.location!.coordinate.longitude }.sorted()
        let median = CLLocation(latitude: latitudes[latitudes.count / 2], longitude: longitudes[longitudes.count / 2])
        let strayCount = usable.reduce(0) { $0 + ($1.location!.distance(from: median) > metres ? 1 : 0) }
        guard strayCount > 0, Double(strayCount) / Double(usable.count) <= maxShare else { return self }
        return filter { sample in
            guard let location = sample.location, location.coordinate.isUsable else { return true }
            return location.distance(from: median) <= metres
        }
    }
}
