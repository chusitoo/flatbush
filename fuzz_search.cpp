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

flatbush::Flatbush<double> createIndex()
{
  flatbush::FlatbushBuilder<double> wBuilder;

  wBuilder.add({ 42, 0, 42, 0 });
  auto wIndex = wBuilder.finish();

  return wIndex;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *iData, size_t iSize)
{
  static auto sIndex = createIndex();

  if (iSize == 32)
  {
    const auto wMinX = *flatbush::detail::bit_cast<const double*>(&iData[0]);
    const auto wMinY = *flatbush::detail::bit_cast<const double*>(&iData[8]);
    const auto wMaxX = *flatbush::detail::bit_cast<const double*>(&iData[16]);
    const auto wMaxY = *flatbush::detail::bit_cast<const double*>(&iData[24]);

    auto wResult = sIndex.search({ wMinX, wMinY, wMaxX, wMaxY });

    if (wMinX <= 42 && wMaxX >= 42 && wMinY <= 0 && wMaxY >= 0)
    {
        assert(wResult.size() == 1);
    }
    else
    {
        assert(wResult.size() == 0);
    }
  }

  return 0;
}
