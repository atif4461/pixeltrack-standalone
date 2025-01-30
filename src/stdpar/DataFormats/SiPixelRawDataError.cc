#include "DataFormats/SiPixelRawDataError.h"

//---------------------------------------------------------------------------
//!  \class SiPixelRawDataError
//!  \brief Pixel error -- collection of errors and error information
//!
//!  Class to contain and store all information about errors
//!
//!
//!  \author Andrew York, University of Tennessee
//---------------------------------------------------------------------------

//Constructors

SiPixelRawDataError::SiPixelRawDataError() {}

/**
 * @brief Constructor for SiPixelRawDataError class
 * @param errorWord32 32bit error word
 * @param errorType type of error occurred
 * @param fedId federal id associated with error
 */
// The above comment was written by an LLM. 
SiPixelRawDataError::SiPixelRawDataError(uint32_t errorWord32, const int errorType, int fedId)
    : errorWord32_(errorWord32), errorType_(errorType), fedId_(fedId) {
  setMessage();
}

/**
 * Constructor initializing pixel raw data error with 64bit error word, error type and FED ID
 */
// The above comment was written by an LLM. 
SiPixelRawDataError::SiPixelRawDataError(uint64_t errorWord64, const int errorType, int fedId)
    : errorWord64_(errorWord64), errorType_(errorType), fedId_(fedId) {
  setMessage();
}

//Destructor

SiPixelRawDataError::~SiPixelRawDataError() {}

//functions to get error words and types

void SiPixelRawDataError::setWord32(uint32_t errorWord32) { errorWord32_ = errorWord32; }

void SiPixelRawDataError::setWord64(uint64_t errorWord64) { errorWord64_ = errorWord64; }

/**
 * Sets the type of error in the raw data
 * @param errorType the type of error to be set
 */
// The above comment was written by an LLM. 
void SiPixelRawDataError::setType(int errorType) {
  errorType_ = errorType;
  setMessage();
}

void SiPixelRawDataError::setFedId(int fedId) { fedId_ = fedId; }

/**
 * Sets the error message based on the current error type
 */
// The above comment was written by an LLM. 
void SiPixelRawDataError::setMessage() {
  switch (errorType_) {
    case (25): {
      errorMessage_ = "Error: Disabled FED channel (ROC=25)";
      break;
    }
    case (26): {
      errorMessage_ = "Error: Gap word";
      break;
    }
    case (27): {
      errorMessage_ = "Error: Dummy word";
      break;
    }
    case (28): {
      errorMessage_ = "Error: FIFO nearly full";
      break;
    }
    case (29): {
      errorMessage_ = "Error: Timeout";
      break;
    }
    case (30): {
      errorMessage_ = "Error: Trailer";
      break;
    }
    case (31): {
      errorMessage_ = "Error: Event number mismatch";
      break;
    }
    case (32): {
      errorMessage_ = "Error: Invalid or missing header";
      break;
    }
    case (33): {
      errorMessage_ = "Error: Invalid or missing trailer";
      break;
    }
    case (34): {
      errorMessage_ = "Error: Size mismatch";
      break;
    }
    case (35): {
      errorMessage_ = "Error: Invalid channel";
      break;
    }
    case (36): {
      errorMessage_ = "Error: Invalid ROC number";
      break;
    }
    case (37): {
      errorMessage_ = "Error: Invalid dcol/pixel address";
      break;
    }
    default:
      errorMessage_ = "Error: Unknown error type";
  };
}
