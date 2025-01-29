#pragma once

#include <s2/s2polygon.h>
#include <s2/s2region_coverer.h>
#include <s2/s2latlng.h>
#include <s2/s2cell_id.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2region_term_indexer.h>
#include <s2/s2cap.h>
#include <s2/s2builder.h>
#include "ids_t.h"
#include "option.h"

class GeoPolygonIndex {
private:
    // Map from cell ID to vector of seq_ids
    std::unordered_map<S2CellId, void*, S2CellIdHash> cellToSeqids;
    // Map of seq_ids to polygons
    std::unordered_map<uint32_t, std::unique_ptr<S2Polygon>> seqidsToPolygons;
    // Coverer configuration
    S2RegionCoverer::Options covererOptions;

public:
    explicit GeoPolygonIndex(int minLevel, int maxLevel, int maxCells) {
        covererOptions.set_min_level(minLevel);
        covererOptions.set_max_level(maxLevel);
        covererOptions.set_max_cells(maxCells);
    }

    ~GeoPolygonIndex() {
        for (auto& item: cellToSeqids) {
            ids_t::destroy_list(item.second);
        }
    }

    //insert seq_ids from mapped S2CellId
    void insertSeqIds(S2CellId value, uint32_t id);

    //get seq_ids from mapped S2CellId
    size_t getSeqIds(S2CellId value, std::vector<uint32_t>& result_ids);

    //remove seq_ids from mapped S2CellId
    void removeSeqIds(S2CellId value, uint32_t id);

    // Add a polygon to the index
    Option<bool> addPolygon(const std::vector<double>& coordinates, uint32_t seq_id);

    // Find all polygons that might contain the given point
    std::vector<uint32_t> findContainingPolygonsRecords(double lat, double lng);

    //remove polygon from index
    void removePolygon(uint32_t seq_id);

    // Get the total number of indexed polygons
    size_t size() const;
};
