#pragma once

#include <functional>
#include <iosfwd>
#include <string>

namespace alliance::storage {

// Replaces a regular file only after its complete payload is durable. A failure
// after rename explicitly reports that the new file is committed, not rolled back.
void WriteAtomicFile(const std::string& path,
                     const std::function<void(std::ostream&)>& serialize);

}  // namespace alliance::storage
