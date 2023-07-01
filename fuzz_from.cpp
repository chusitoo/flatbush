/*
MIT License

Copyright (c) 2023 Alex Emirov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "flatbush.h"

#include <cassert>

template <typename ArrayType>
flatbush::Flatbush<ArrayType> createIndex(uint32_t iNumItems, uint16_t iNodeSize)
{
  flatbush::FlatbushBuilder<ArrayType> wBuilder(iNumItems, iNodeSize);

  auto wSize = static_cast<size_t>(iNumItems);
  for (size_t wIdx = 0; wIdx < wSize; ++wIdx)
  {
    auto coord = static_cast<ArrayType>(wIdx);
    wBuilder.add({ coord, coord, coord, coord });
  }
  auto wIndex = wBuilder.finish();

  return wIndex;
}

template <typename ArrayType>
int from(const uint8_t *iData, size_t iSize)
{
  if (iSize < flatbush::gHeaderByteSize) return 0;
  if (iData[0] != flatbush::gValidityFlag) return 0;
  if ((iData[1] >> 4) != flatbush::gVersion) return 0;
  if ((iData[1] & 0x0f) != flatbush::detail::arrayTypeIndex<ArrayType>()) return 0;
  const auto wNodeSize = *flatbush::detail::bit_cast<uint16_t*>(&iData[2]);
  if (wNodeSize < 2) return 0;

  const auto wNumItems = *flatbush::detail::bit_cast<uint32_t*>(&iData[4]);
  const auto& wLevelBounds = flatbush::detail::calculateNumNodesPerLevel(wNumItems, wNodeSize);
  const auto wNumNodes = wLevelBounds.empty() ? wNumItems : wLevelBounds.back();
  const auto wIndicesByteSize = wNumNodes * ((wNumNodes >= 16384) ? sizeof(uint32_t) : sizeof(uint16_t));
  const auto wNodesByteSize = wNumNodes * sizeof(flatbush::Box<ArrayType>);
  const auto wSize = flatbush::gHeaderByteSize + wNodesByteSize + wIndicesByteSize;
  if (wSize != iSize) return 0;

  auto wIndex = flatbush::FlatbushBuilder<ArrayType>::from(iData, iSize);

  assert(wIndex.data().size() == iSize);
  assert(wIndex.nodeSize() == wNodeSize);
  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.indexSize() == wNumNodes);

  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *iData, size_t iSize)
{
  from<int8_t>(iData, iSize);
  from<uint8_t>(iData, iSize);
  from<int16_t>(iData, iSize);
  from<uint16_t>(iData, iSize);
  from<int32_t>(iData, iSize);
  from<uint32_t>(iData, iSize);
  from<float>(iData, iSize);
  from<double>(iData, iSize);
  return 0;
}
