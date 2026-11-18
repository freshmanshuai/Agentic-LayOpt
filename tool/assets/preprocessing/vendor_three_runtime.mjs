import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const sourceRoot = path.join(repoRoot, "node_modules", "three");
const vendorRoot = path.join(repoRoot, "web_frontend", "vendor", "three");

const files = [
  ["build/three.module.js", "three.module.js"],
  ["build/three.core.js", "three.core.js"],
  ["examples/jsm/controls/OrbitControls.js", "addons/controls/OrbitControls.js"],
  ["examples/jsm/loaders/GLTFLoader.js", "addons/loaders/GLTFLoader.js"],
  ["examples/jsm/libs/meshopt_decoder.module.js", "addons/libs/meshopt_decoder.module.js"],
  ["examples/jsm/utils/BufferGeometryUtils.js", "addons/utils/BufferGeometryUtils.js"],
  ["examples/jsm/utils/SkeletonUtils.js", "addons/utils/SkeletonUtils.js"],
];

for (const [source, destination] of files) {
  const outputPath = path.join(vendorRoot, destination);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await copyFile(path.join(sourceRoot, source), outputPath);
}

console.log(`Vendored Three.js ${files.length} files to ${vendorRoot}`);
