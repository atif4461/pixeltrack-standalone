#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "Source.h"

namespace {
/**
 * Reads raw data from input stream into collection of FED raw data objects
 * @param is Input stream containing raw data
 * @param nfeds Number of FEDs to read from input stream
 * @return Collection of FED raw data objects populated from input stream
 */
// The above comment was written by an LLM. 
  FEDRawDataCollection readRaw(std::ifstream &is, unsigned int nfeds) {
    FEDRawDataCollection rawCollection;
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId;
      is.read(reinterpret_cast<char *>(&fedId), sizeof(unsigned int));
      unsigned int fedSize;
      is.read(reinterpret_cast<char *>(&fedSize), sizeof(unsigned int));
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize);
      is.read(reinterpret_cast<char *>(rawData.data()), fedSize);
    }
    return rawCollection;
  }

}  // namespace

namespace edm {
/**
 * Constructor for Source class
 * @param maxEvents maximum number of events to process
 * @param runForMinutes runtime duration in minutes
 * @param reg product registry reference
 * @param datadir data directory path
 * @param validation flag indicating whether validation is enabled
 */
// The above comment was written by an LLM. 
  Source::Source(
      int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &datadir, bool validation)
      : maxEvents_(maxEvents),
        runForMinutes_(runForMinutes),
        rawToken_(reg.produces<FEDRawDataCollection>()),
        validation_(validation) {
    std::ifstream in_raw(datadir / "raw.bin", std::ios::binary);
    std::ifstream in_digiclusters;
    std::ifstream in_tracks;
    std::ifstream in_vertices;

    if (validation_) {
      digiClusterToken_ = reg.produces<DigiClusterCount>();
      trackToken_ = reg.produces<TrackCount>();
      vertexToken_ = reg.produces<VertexCount>();

      in_digiclusters = std::ifstream(datadir / "digicluster.bin", std::ios::binary);
      in_tracks = std::ifstream(datadir / "tracks.bin", std::ios::binary);
      in_vertices = std::ifstream(datadir / "vertices.bin", std::ios::binary);
      in_digiclusters.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in_tracks.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in_vertices.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    }

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

      raw_.emplace_back(readRaw(in_raw, nfeds));

      if (validation_) {
        unsigned int nm, nd, nc, nt, nv;
        in_digiclusters.read(reinterpret_cast<char *>(&nm), sizeof(unsigned int));
        in_digiclusters.read(reinterpret_cast<char *>(&nd), sizeof(unsigned int));
        in_digiclusters.read(reinterpret_cast<char *>(&nc), sizeof(unsigned int));
        in_tracks.read(reinterpret_cast<char *>(&nt), sizeof(unsigned int));
        in_vertices.read(reinterpret_cast<char *>(&nv), sizeof(unsigned int));
        digiclusters_.emplace_back(nm, nd, nc);
        tracks_.emplace_back(nt);
        vertices_.emplace_back(nv);
      }

      // next event
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    }

    if (validation_) {
      assert(raw_.size() == digiclusters_.size());
      assert(raw_.size() == tracks_.size());
      assert(raw_.size() == vertices_.size());
    }

    if (runForMinutes_ < 0 and maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }
  }

/**
 * Reconfigures the source with new parameters.
 * @param maxEvents The maximum number of events to process.
 * @param runForMinutes The duration in minutes to run the source.
 */
// The above comment was written by an LLM. 
  void Source::reconfigure(int maxEvents, int runForMinutes) {
    std::scoped_lock lock(timeMutex_);
    maxEvents_ = maxEvents;
    runForMinutes_ = runForMinutes;
    numEventsTimeLastCheck_ = 0;
    shouldStop_ = false;
    numEvents_ = 0;
  }

/**
 * Starts the processing of the source data when run time is nonnegative
 */
// The above comment was written by an LLM. 
  void Source::startProcessing() {
    if (runForMinutes_ >= 0) {
      startTime_ = std::chrono::steady_clock::now();
    }
  }

/**
 * @brief Produces an event based on the provided stream ID and product registry.
 * 
 * @param streamId The identifier of the stream.
 * @param reg The product registry.
 * @return A unique pointer to the produced event or nullptr if production should stop.
 */
// The above comment was written by an LLM. 
  std::unique_ptr<Event> Source::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return nullptr;
    }

    const int old = numEvents_.fetch_add(1);
    const int iev = old + 1;
    if (runForMinutes_ < 0) {
      if (old >= maxEvents_) {
        shouldStop_ = true;
        --numEvents_;
        return nullptr;
      }
    } else {
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
        std::scoped_lock lock(timeMutex_);
        // if some other thread beat us, no need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
          auto processingTime = std::chrono::steady_clock::now() - startTime_;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / raw_.size()) * raw_.size();
        }
        if (shouldStop_) {
          --numEvents_;
          return nullptr;
        }
      }
    }
    auto ev = std::make_unique<Event>(streamId, iev, reg);
    const int index = old % raw_.size();

    ev->emplace(rawToken_, raw_[index]);
    if (validation_) {
      ev->emplace(digiClusterToken_, digiclusters_[index]);
      ev->emplace(trackToken_, tracks_[index]);
      ev->emplace(vertexToken_, vertices_[index]);
    }

    return ev;
  }
}  // namespace edm
