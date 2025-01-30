/** \file
 *
 *  \author N. Amapane - CERN, R. Mommsen - FNAL
 */

#include "DataFormats/FEDHeader.h"
#include "DataFormats/fed_header.h"

FEDHeader::FEDHeader(const unsigned char* header) : theHeader(reinterpret_cast<const fedh_t*>(header)) {}

FEDHeader::~FEDHeader() {}

uint8_t FEDHeader::triggerType() const { return FED_EVTY_EXTRACT(theHeader->eventid); }

uint32_t FEDHeader::lvl1ID() const { return FED_LVL1_EXTRACT(theHeader->eventid); }

uint16_t FEDHeader::bxID() const { return FED_BXID_EXTRACT(theHeader->sourceid); }

uint16_t FEDHeader::sourceID() const { return FED_SOID_EXTRACT(theHeader->sourceid); }

uint8_t FEDHeader::version() const { return FED_VERSION_EXTRACT(theHeader->sourceid); }

bool FEDHeader::moreHeaders() const { return (FED_MORE_HEADERS_EXTRACT(theHeader->sourceid) != 0); }

/**
 * Sets the values of the FED header structure from the given parameters.
 * @param[in] header     Pointer to the unsigned char array containing the header data
 * @param[in] triggerType Type of trigger
 * @param[in] lvl1ID      Level one ID
 * @param[in] bxID       Bunch crossing ID
 * @param[in] sourceID   Source identifier
 * @param[in] version    Version number
 * @param[in] moreHeaders Flag indicating whether there are more headers
 */
// The above comment was written by an LLM. 
void FEDHeader::set(unsigned char* header,
                    uint8_t triggerType,
                    uint32_t lvl1ID,
                    uint16_t bxID,
                    uint16_t sourceID,
                    uint8_t version,
                    bool moreHeaders) {
  // FIXME: should check that input ranges are OK!!!
  fedh_t* h = reinterpret_cast<fedh_t*>(header);
  h->eventid = (FED_SLINK_START_MARKER << FED_HCTRLID_SHIFT) | ((triggerType << FED_EVTY_SHIFT) & FED_EVTY_MASK) |
               ((lvl1ID << FED_LVL1_SHIFT) & FED_LVL1_MASK);

  h->sourceid = ((bxID << FED_BXID_SHIFT) & FED_BXID_MASK) | ((sourceID << FED_SOID_SHIFT) & FED_SOID_MASK) |
                ((version << FED_VERSION_SHIFT) & FED_VERSION_MASK);

  if (moreHeaders)
    h->sourceid |= (FED_MORE_HEADERS_WIDTH << FED_MORE_HEADERS_SHIFT);
}

bool FEDHeader::check() const { return (FED_HCTRLID_EXTRACT(theHeader->eventid) == FED_SLINK_START_MARKER); }

const uint32_t FEDHeader::length = sizeof(fedh_t);
