#include <flatbush.h>

int main() {
  flatbush::FlatbushBuilder<double> wBuilder(1);
  wBuilder.add({ 0.0, 0.0, 1.0, 1.0 });
  const auto wIndex = wBuilder.finish();

  return wIndex.search({ 0.0, 0.0, 1.0, 1.0 }).size() == 1UL ? 0 : 1;
}