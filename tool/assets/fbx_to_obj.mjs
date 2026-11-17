import fs from "node:fs";
import path from "node:path";
import { once } from "node:events";

import { Box3, Vector3 } from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

const [sourcePath, outputPath] = process.argv.slice(2);

if (!sourcePath || !outputPath) {
  console.error("Usage: node tool/assets/fbx_to_obj.mjs <source.fbx> <output.obj>");
  process.exit(2);
}

const buffer = fs.readFileSync(sourcePath);
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const object = new FBXLoader().parse(arrayBuffer, path.dirname(sourcePath) + path.sep);
object.updateMatrixWorld(true);

let meshObjectCount = 0;
let vertices = 0;
let faces = 0;
object.traverse((node) => {
  if (!node.isMesh || !node.geometry) {
    return;
  }
  meshObjectCount += 1;
  const position = node.geometry.getAttribute("position");
  if (position) {
    vertices += position.count;
  }
  if (node.geometry.index) {
    faces += Math.floor(node.geometry.index.count / 3);
  } else if (position) {
    faces += Math.floor(position.count / 3);
  }
});

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
await writeObjStream(object, outputPath);

const box = new Box3().setFromObject(object);
const size = new Vector3();
box.getSize(size);

console.log(JSON.stringify({
  object_count: object.children.length,
  mesh_object_count: meshObjectCount,
  vertices,
  faces,
  bounds: {
    min: { x: box.min.x, y: box.min.y, z: box.min.z },
    max: { x: box.max.x, y: box.max.y, z: box.max.z },
    size: { x: size.x, y: size.y, z: size.z },
  },
}));

async function writeObjStream(object, outputPath) {
  const stream = fs.createWriteStream(outputPath, { encoding: "utf8" });
  let vertexOffset = 0;
  const point = new Vector3();

  await write(stream, `# Exported from ${path.basename(sourcePath)}\n`);
  for (const mesh of collectMeshes(object)) {
    const geometry = mesh.geometry;
    const position = geometry.getAttribute("position");
    if (!position) {
      continue;
    }
    await write(stream, `o ${sanitizeName(mesh.name || "mesh")}\n`);
    for (let i = 0; i < position.count; i += 1) {
      point.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
      await write(stream, `v ${point.x} ${point.y} ${point.z}\n`);
    }

    if (geometry.index) {
      const index = geometry.index;
      for (let i = 0; i + 2 < index.count; i += 3) {
        const a = index.getX(i) + 1 + vertexOffset;
        const b = index.getX(i + 1) + 1 + vertexOffset;
        const c = index.getX(i + 2) + 1 + vertexOffset;
        await write(stream, `f ${a} ${b} ${c}\n`);
      }
    } else {
      for (let i = 0; i + 2 < position.count; i += 3) {
        const a = i + 1 + vertexOffset;
        const b = i + 2 + vertexOffset;
        const c = i + 3 + vertexOffset;
        await write(stream, `f ${a} ${b} ${c}\n`);
      }
    }
    vertexOffset += position.count;
  }
  stream.end();
  await once(stream, "finish");
}

function collectMeshes(object) {
  const meshes = [];
  object.traverse((node) => {
    if (node.isMesh && node.geometry) {
      meshes.push(node);
    }
  });
  return meshes;
}

async function write(stream, text) {
  if (!stream.write(text)) {
    await once(stream, "drain");
  }
}

function sanitizeName(value) {
  return String(value).replace(/\s+/g, "_").replace(/[^\w.-]/g, "_") || "mesh";
}
