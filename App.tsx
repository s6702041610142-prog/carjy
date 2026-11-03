/**
 * 3D arcade racer: expo-gl + three.js, tuned for the Samsung Galaxy A14.
 *
 * Install: npx expo install expo-gl expo-sensors expo-keep-awake expo-navigation-bar expo-build-properties three
 *          npm i -D @types/three
 *
 * Karts are simulated in track space (s = distance along the circuit, d = lateral
 * offset) instead of free 2D physics. Corners push karts outward, so steering still
 * matters, while laps and ranking reduce to comparing a single number.
 */
import { ExpoWebGLRenderingContext, GLView } from 'expo-gl';
import { useKeepAwake } from 'expo-keep-awake';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { NavigationBar } from 'expo-navigation-bar';
import { StatusBar } from 'expo-status-bar';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Image, LayoutChangeEvent, PixelRatio, Platform, Pressable, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import * as THREE from 'three';

import { CAR_MODEL, CAR_POSITION_SCALE, CAR_WHEEL_HD_MESH, CAR_WHEEL_MESH, CAR_WHEEL_POSITIONS, EncodedMesh } from './carModel';

// Colours are authored and lit in display space: this skips the per-pixel sRGB
// conversion, which is a measurable saving on a Mali G52/G57.
THREE.ColorManagement.enabled = false;

/* -------------------------------------------------------------------------- */
/* Tuning                                                                     */
/* -------------------------------------------------------------------------- */

const LAPS = 3;

const DEVICE_PIXEL_RATIO = PixelRatio.get();
const IS_WEB = Platform.OS === 'web';

type Quality = 'low' | 'medium' | 'high' | 'ultra';
type CarLod = 'player' | 'cpu' | 'ultra';

interface QualitySettings {
  label: string;
  note: string;
  pixelRatio: number;
  playerLod: CarLod;
  cpuLod: CarLod;
  wheelHd: boolean;
  palms: number;
  bushes: number;
  clouds: boolean;
  shadowSize: number; // 0 keeps soft blob shadows only
  realReflections: boolean; // physically based paint with an environment map
}

/** "medium" is the Galaxy A14 baseline; "low" leaves headroom for sustained or weaker play. */
const QUALITY: Record<Quality, QualitySettings> = {
  low: { label: 'Low', note: 'Smoothest, saves battery', pixelRatio: 1, playerLod: 'cpu', cpuLod: 'cpu', wheelHd: false, palms: 0, bushes: 0, clouds: false, shadowSize: 0, realReflections: false },
  medium: { label: 'Medium', note: 'Balanced', pixelRatio: 1.25, playerLod: 'player', cpuLod: 'cpu', wheelHd: false, palms: 10, bushes: 0, clouds: true, shadowSize: 0, realReflections: false },
  high: { label: 'High', note: 'Real shadows, more trees', pixelRatio: 1.5, playerLod: 'player', cpuLod: 'player', wheelHd: false, palms: 24, bushes: 70, clouds: true, shadowSize: 1024, realReflections: false },
  ultra: { label: 'Ultra', note: 'Best looking, may stutter on phones', pixelRatio: 2, playerLod: 'ultra', cpuLod: IS_WEB ? 'ultra' : 'player', wheelHd: true, palms: 40, bushes: 160, clouds: true, shadowSize: 2048, realReflections: IS_WEB },
};

/** React Native has no window.devicePixelRatio, so PixelRatio supplies the clamp. */
function renderPixelRatio(quality: Quality) {
  const wanted = QUALITY[quality].pixelRatio;
  return IS_WEB ? wanted : Math.min(DEVICE_PIXEL_RATIO, wanted);
}

const REMOTE_PORT = 8090;

const SENSOR_LERP = 0.08;
const SENSOR_INTERVAL_MS = 16;
const TILT_DEADZONE = 0.04; // radians
const TILT_FULL_LOCK = 0.38; // radians, roughly 22 degrees

const TRACK_SEGMENTS = 120;
const ROAD_HALF = 7;
const KERB_OUTER = 8;
const WALL = 10.5;
const KART_WALL_LIMIT = WALL - 1.3;
const HILL_OUTER = 27;
const ROAD_TILE_METRES = 24;
const SKY_RADIUS = 300;
const SKY_HORIZON = 0xd5e9f6;
const SKY_ZENITH = 0x2f83d6;

const PLAYER_TOP_SPEED = 32; // m/s
const CPU_TOP_SPEEDS = [33.4, 32.4, 31.4];
const STEER_RATE = 13; // lateral m/s at full lock
const CORNER_PUSH = 0.25; // fraction of centripetal acceleration felt as outward drift
const OFFROAD_FACTOR = 0.55;
const MUSHROOM_MULTIPLIER = 1.5;
const MUSHROOM_SECONDS = 2;
const PAD_MULTIPLIER = 1.3;
const PAD_SECONDS = 1;
const SPIN_SECONDS = 1;
const SPIN_SPEED_FACTOR = 0.35;

const BOX_COUNT = 8;
const BOX_RESPAWN_SECONDS = 5;
const BANANA_COUNT = 5;
const SHELL_COUNT = 3;
const SHELL_SPEED = 60;
const SHELL_LIFE_SECONDS = 3.5;
const PARTICLE_COUNT = 64;
const PARTICLE_LIFE_SECONDS = 0.4;
const WAYPOINT_COUNT = 32;
const PAD_FRACTIONS = [0.12, 0.38, 0.62, 0.9];

type ItemKind = 'mushroom' | 'banana' | 'shell';
type Phase = 'countdown' | 'racing' | 'finished';

const ITEM_KINDS: ItemKind[] = ['mushroom', 'banana', 'shell'];
const ITEM_ICON: Record<ItemKind, string> = { mushroom: '🍄', banana: '🍌', shell: '🐢' };
const PLACE_LABEL = ['1st', '2nd', '3rd', '4th'];
const KART_SHADOWS = 4;

/* -------------------------------------------------------------------------- */
/* Track: a closed polyline with uniform arc-length samples                   */
/* -------------------------------------------------------------------------- */

interface TrackPoint {
  x: number;
  z: number;
  tx: number;
  tz: number;
  k: number; // curvature; positive when the road bends right
}

const CONTROL_POINTS: [number, number][] = [
  [-30, -118], [40, -118], [100, -108], [122, -68], [92, -32], [48, -18], [46, 22], [84, 52],
  [90, 96], [48, 122], [-18, 112], [-42, 66], [-92, 56], [-124, 8], [-112, -62], [-78, -108],
];

function buildTrack() {
  const curve = new THREE.CatmullRomCurve3(
    CONTROL_POINTS.map(([x, z]) => new THREE.Vector3(x, 0, z)),
    true,
    'centripetal',
  );
  const points = curve.getSpacedPoints(TRACK_SEGMENTS);
  const n = TRACK_SEGMENTS;
  const px = new Float32Array(n);
  const pz = new Float32Array(n);
  const tx = new Float32Array(n);
  const tz = new Float32Array(n);
  const k = new Float32Array(n);
  let length = 0;
  for (let i = 0; i < n; i++) {
    px[i] = points[i].x;
    pz[i] = points[i].z;
    length += points[i].distanceTo(points[i + 1]);
  }
  const segment = length / n;
  for (let i = 0; i < n; i++) {
    const a = (i + n - 1) % n;
    const b = (i + 1) % n;
    const dx = px[b] - px[a];
    const dz = pz[b] - pz[a];
    const len = Math.hypot(dx, dz);
    tx[i] = dx / len;
    tz[i] = dz / len;
  }
  for (let i = 0; i < n; i++) {
    const a = (i + n - 1) % n;
    const b = (i + 1) % n;
    // Right vector is (-tz, tx).
    k[i] = ((tx[b] - tx[a]) * -tz[i] + (tz[b] - tz[a]) * tx[i]) / (2 * segment);
  }
  // Light smoothing so corner forces ramp in rather than stepping per segment.
  for (let pass = 0; pass < 2; pass++) {
    const copy = k.slice();
    for (let i = 0; i < n; i++) {
      k[i] = (copy[(i + n - 1) % n] + 2 * copy[i] + copy[(i + 1) % n]) / 4;
    }
  }
  return { n, length, segment, px, pz, tx, tz, k };
}

const TRACK = buildTrack();

function sampleTrack(s: number, out: TrackPoint) {
  let u = s % TRACK.length;
  if (u < 0) u += TRACK.length;
  const f = u / TRACK.segment;
  const i = Math.floor(f) % TRACK.n;
  const j = (i + 1) % TRACK.n;
  const a = f - Math.floor(f);
  out.x = TRACK.px[i] + (TRACK.px[j] - TRACK.px[i]) * a;
  out.z = TRACK.pz[i] + (TRACK.pz[j] - TRACK.pz[i]) * a;
  const tx = TRACK.tx[i] + (TRACK.tx[j] - TRACK.tx[i]) * a;
  const tz = TRACK.tz[i] + (TRACK.tz[j] - TRACK.tz[i]) * a;
  const len = Math.sqrt(tx * tx + tz * tz) || 1;
  out.tx = tx / len;
  out.tz = tz / len;
  out.k = TRACK.k[i] + (TRACK.k[j] - TRACK.k[i]) * a;
}

/** Shortest signed distance from b to a around the loop. */
function wrapDelta(a: number, b: number) {
  const L = TRACK.length;
  let delta = (a - b) % L;
  if (delta > L / 2) delta -= L;
  else if (delta < -L / 2) delta += L;
  return delta;
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

/* -------------------------------------------------------------------------- */
/* Geometry: everything is merged into flat-shaded, vertex-coloured buffers   */
/* -------------------------------------------------------------------------- */

class MeshBuilder {
  private positions: number[] = [];
  private normals: number[] = [];
  private colors: number[] = [];
  private uvs: number[] = [];

  /** Bakes a three.js primitive in, with face normals for a crisp faceted look. */
  add(primitive: THREE.BufferGeometry, color: number, matrix?: THREE.Matrix4) {
    const g = primitive.index ? primitive.toNonIndexed() : primitive;
    if (matrix) g.applyMatrix4(matrix);
    g.computeVertexNormals();
    this.append(g, () => color);
    primitive.dispose();
    g.dispose();
    return this;
  }

  /** Keeps the geometry's existing smooth normals and colours each vertex by position. */
  addSmooth(primitive: THREE.BufferGeometry, colorAt: (x: number, y: number, z: number) => number) {
    const g = primitive.index ? primitive.toNonIndexed() : primitive;
    this.append(g, colorAt);
    primitive.dispose();
    g.dispose();
    return this;
  }

  private append(g: THREE.BufferGeometry, colorAt: (x: number, y: number, z: number) => number) {
    const p = g.getAttribute('position');
    const n = g.getAttribute('normal');
    for (let i = 0; i < p.count; i++) {
      const [r, gr, bl] = rgb(colorAt(p.getX(i), p.getY(i), p.getZ(i)));
      this.positions.push(p.getX(i), p.getY(i), p.getZ(i));
      this.normals.push(n.getX(i), n.getY(i), n.getZ(i));
      this.colors.push(r, gr, bl);
      this.uvs.push(0, 0);
    }
  }

  /** Adds a counter-clockwise quad a-b-c-d, with one colour or one per corner. */
  quad(a: number[], b: number[], c: number[], d: number[], color: number | number[], uv?: number[]) {
    const cs = typeof color === 'number' ? [color, color, color, color] : color;
    const t = uv ?? [0, 0, 0, 0, 0, 0, 0, 0];
    this.triangle(a, b, c, cs[0], cs[1], cs[2], t[0], t[1], t[2], t[3], t[4], t[5]);
    this.triangle(a, c, d, cs[0], cs[2], cs[3], t[0], t[1], t[4], t[5], t[6], t[7]);
  }

  triangle(a: number[], b: number[], c: number[], ca: number, cb: number, cc: number, ua = 0, va = 0, ub = 0, vb = 0, uc = 0, vc = 0) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    const corners: [number[], number, number, number][] = [[a, ca, ua, va], [b, cb, ub, vb], [c, cc, uc, vc]];
    for (const [v, color, tu, tv] of corners) {
      const [r, gr, bl] = rgb(color);
      this.positions.push(v[0], v[1], v[2]);
      this.normals.push(nx, ny, nz);
      this.colors.push(r, gr, bl);
      this.uvs.push(tu, tv);
    }
  }

  build(withUV = false) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.colors, 3));
    if (withUV) g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2));
    return g;
  }
}

function rgb(hex: number): [number, number, number] {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

function transform(x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) {
  return new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)),
    new THREE.Vector3(sx, sy, sz),
  );
}

function trackVertex(s: number, d: number, y: number) {
  const p: TrackPoint = { x: 0, z: 0, tx: 0, tz: 1, k: 0 };
  sampleTrack(s, p);
  return [p.x - p.tz * d, y, p.z + p.tx * d];
}

function triangleCount(g: THREE.BufferGeometry) {
  return g.getAttribute('position').count / 3;
}

function seededRandom(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function distanceToCentreline(x: number, z: number) {
  let best = Infinity;
  for (let i = 0; i < TRACK.n; i++) {
    const dist = Math.hypot(TRACK.px[i] - x, TRACK.pz[i] - z);
    if (dist < best) best = dist;
  }
  return best;
}

/** Picks points beside the track; never on the road, whatever the track shape. */
function scatterBesideTrack(random: () => number, count: number, minOffset: number, maxOffset: number) {
  const out: [number, number][] = [];
  for (let attempt = 0; attempt < count * 40 && out.length < count; attempt++) {
    const s = random() * TRACK.length;
    const offset = (random() < 0.5 ? -1 : 1) * (minOffset + random() * (maxOffset - minOffset));
    const [x, , z] = trackVertex(s, offset, 0);
    if (distanceToCentreline(x, z) >= minOffset - 0.5) out.push([x, z]);
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Procedural textures: generated at start-up, so the app ships no image files */
/* -------------------------------------------------------------------------- */

function createTexture(
  width: number,
  height: number,
  anisotropy: number,
  paint: (x: number, y: number, out: number[]) => void,
) {
  const data = new Uint8Array(width * height * 4);
  const out = [0, 0, 0, 255];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[3] = 255;
      paint(x, y, out);
      const i = (y * width + x) * 4;
      data[i] = clamp(out[0], 0, 255);
      data[i + 1] = clamp(out[1], 0, 255);
      data[i + 2] = clamp(out[2], 0, 255);
      data[i + 3] = clamp(out[3], 0, 255);
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = anisotropy;
  texture.needsUpdate = true;
  return texture;
}

/** One 24 m tile of road: worn asphalt, tyre marks, tar seams, lines and a dusty edge. */
function createRoadTexture(anisotropy: number) {
  const W = 256;
  const H = 512;
  const random = seededRandom(11);
  const TAU = Math.PI * 2;
  return createTexture(W, H, anisotropy, (x, y, o) => {
    const d = ((x + 0.5) / W) * (2 * KERB_OUTER) - KERB_OUTER;
    const v = (y + 0.5) / H;
    const metres = v * ROAD_TILE_METRES;
    const grain = random();
    let tone = 76 + (grain - 0.5) * 26 + Math.sin(v * TAU * 3 + d * 0.7) * 5 + Math.sin(v * TAU * 7 - d * 1.9) * 3;
    // Tyre lines, swerving a little and repeating exactly once per tile.
    const lane1 = d - 2.4 - Math.sin(v * TAU) * 0.35;
    const lane2 = d + 2.4 - Math.sin(v * TAU + 1.5) * 0.35;
    const lane3 = d - 4.6 - Math.sin(v * TAU * 2) * 0.2;
    tone -= 15 * Math.exp(-lane1 * lane1 * 3) + 15 * Math.exp(-lane2 * lane2 * 3) + 8 * Math.exp(-lane3 * lane3 * 5);
    // Repaired cracks.
    if (Math.abs(d - (1.2 + Math.sin(v * TAU * 2) * 3.4)) < 0.07) tone -= 22;
    if (Math.abs(d + (3.6 + Math.sin(v * TAU * 3 + 1) * 1.6)) < 0.05) tone -= 16;
    let r = tone;
    let g = tone;
    let b = tone + 3;
    const side = Math.abs(d);
    if (side > 6.5 && side < 6.8) {
      r = g = b = 205 + (grain - 0.5) * 30; // edge line
    } else if (side < 0.11 && metres % 12 < 5) {
      r = g = b = 195 + (grain - 0.5) * 34; // centre dash
    } else if (side >= 6.95) {
      // Dust where asphalt meets the verge.
      const dust = clamp((side - 6.95) / 1.05, 0, 1) * (0.65 + grain * 0.35);
      r += (150 - r) * dust;
      g += (128 - g) * dust;
      b += (92 - b) * dust;
    }
    o[0] = r;
    o[1] = g;
    o[2] = b;
  });
}

/** A neutral blotchy tile, multiplied by vertex colours for grass, gravel and earth. */
function createGroundTexture(anisotropy: number) {
  const S = 128;
  const random = seededRandom(5);
  const TAU = Math.PI * 2;
  return createTexture(S, S, anisotropy, (x, y, o) => {
    const u = x / S;
    const v = y / S;
    const blotch =
      Math.sin(u * TAU * 2 + Math.sin(v * TAU * 3) * 1.2) * 0.5 +
      Math.sin(v * TAU * 5 + Math.sin(u * TAU * 2) * 1.4) * 0.3;
    const value = 196 + blotch * 26 + (random() - 0.5) * 62;
    o[0] = value * 1.02;
    o[1] = value;
    o[2] = value * 0.94;
  });
}

function createKerbTexture() {
  return createTexture(32, 4, 1, (x, _y, o) => {
    const red = x < 16;
    o[0] = red ? 208 : 236;
    o[1] = red ? 44 : 236;
    o[2] = red ? 38 : 232;
  });
}

/** Eight metres of dark concrete wall, with a band of colour and panel joints. */
function createBarrierTexture() {
  const W = 128;
  const H = 32;
  const random = seededRandom(9);
  const BANDS = [0xd8332a, 0xf1f1f1, 0x1f6fd0, 0xf1b90f];
  return createTexture(W, H, 1, (x, y, o) => {
    const panel = Math.floor(x / 32);
    const band = rgb(BANDS[panel]);
    const tone = 64 + random() * 16 + (y < 3 ? 26 : 0);
    const inBand = y >= 11 && y < 20;
    const joint = x % 32 < 1;
    o[0] = joint ? 34 : inBand ? band[0] * 255 : tone;
    o[1] = joint ? 34 : inBand ? band[1] * 255 : tone;
    o[2] = joint ? 36 : inBand ? band[2] * 255 : tone + 2;
  });
}

/** Yellow and black corner marker, pointing towards +u. */
function createChevronTexture() {
  const W = 64;
  const H = 32;
  return createTexture(W, H, 1, (x, y, o) => {
    const phase = (x + Math.abs(y - H / 2) * 1.1) % 32;
    const edge = x < 2 || x >= W - 2 || y < 2 || y >= H - 2;
    const black = edge || phase < 14;
    o[0] = black ? 18 : 246;
    o[1] = black ? 18 : 194;
    o[2] = black ? 20 : 26;
  });
}

function createBlobTexture() {
  const S = 32;
  return createTexture(S, S, 1, (x, y, o) => {
    const dist = Math.hypot(x + 0.5 - S / 2, y + 0.5 - S / 2) / (S / 2);
    const fade = clamp(1 - dist, 0, 1);
    o[0] = o[1] = o[2] = 255;
    o[3] = fade * fade * (3 - 2 * fade) * 255;
  });
}

function createCloudTexture() {
  const S = 64;
  const random = seededRandom(31);
  const puffs: [number, number, number][] = [];
  for (let i = 0; i < 16; i++) {
    const a = random() * Math.PI * 2;
    const r = random() * 0.62;
    puffs.push([0.5 + Math.cos(a) * r, 0.5 + Math.sin(a) * r * 0.42, 0.1 + random() * 0.16]);
  }
  const texture = createTexture(S, S, 1, (x, y, o) => {
    let alpha = 0;
    for (let i = 0; i < puffs.length; i++) {
      const [px, py, radius] = puffs[i];
      const dist = Math.hypot(x / S - px, y / S - py) / radius;
      alpha = Math.max(alpha, clamp(1 - dist * dist, 0, 1));
    }
    o[0] = o[1] = o[2] = 255;
    // Fade to nothing well before the edge of the quad.
    const edge = clamp((1 - Math.hypot((x / S - 0.5) * 2, (y / S - 0.5) * 2)) * 3, 0, 1);
    o[3] = alpha * edge * 215;
  });
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

/* -------------------------------------------------------------------------- */
/* World geometry                                                             */
/* -------------------------------------------------------------------------- */

/** Hill height beside the track. Whole-number frequencies make it close around the lap. */
function hillHeight(s: number, d: number) {
  const a = (s / TRACK.length) * Math.PI * 2;
  const side = d > 0 ? 1 : -1;
  const amplitude =
    5.5 + Math.sin(a * 3 + side) * 2.6 + Math.sin(a * 7 + side * 2.2) * 1.6 + Math.sin(a * 13 - side) * 0.8;
  const t = clamp((Math.abs(d) - WALL) / (HILL_OUTER - WALL), 0, 1);
  return amplitude * t * t * (3 - 2 * t);
}

function hillOffset(sampleIndex: number, side: number, fraction: number) {
  const k = TRACK.k[(sampleIndex + TRACK.n) % TRACK.n];
  // On the inside of a corner the offset must stay short of the centre of curvature.
  const limit = k * side > 0.001 ? Math.min(HILL_OUTER, 0.8 / Math.abs(k)) : HILL_OUTER;
  return side * (WALL + (Math.max(limit, WALL + 2) - WALL) * fraction);
}

function mixColor(a: number, b: number, t: number) {
  const ca = rgb(a);
  const cb = rgb(b);
  const channel = (i: number) => Math.round((ca[i] + (cb[i] - ca[i]) * t) * 255);
  return (channel(0) << 16) | (channel(1) << 8) | channel(2);
}

function buildSkyGeometry() {
  const b = new MeshBuilder();
  const random = seededRandom(3);

  const dome = new THREE.SphereGeometry(SKY_RADIUS, 12, 6).toNonIndexed();
  const domeVertices = dome.getAttribute('position');
  for (let i = 0; i < domeVertices.count; i += 3) {
    const corner = (n: number) => {
      const y = domeVertices.getY(i + n);
      return [domeVertices.getX(i + n), y, domeVertices.getZ(i + n)];
    };
    const tint = (v: number[]) => mixColor(SKY_HORIZON, SKY_ZENITH, clamp(v[1] / (SKY_RADIUS * 0.72), 0, 1));
    // Reversed winding: the dome is seen from the inside.
    const p0 = corner(0);
    const p1 = corner(2);
    const p2 = corner(1);
    b.triangle(p0, p1, p2, tint(p0), tint(p1), tint(p2));
  }
  dome.dispose();

  return b.build();
}

/** The far scenery is a picture: hazy ridges, a tree line and a resort on one open cylinder. */
function createBackdropTexture() {
  const W = 1024;
  const H = 128;
  const TAU = Math.PI * 2;
  const random = seededRandom(13);
  const ridge = (u: number, f: number, seed: number) =>
    Math.sin(u * TAU * f + seed) * 0.5 + Math.sin(u * TAU * (f * 2 + 1) + seed * 2.3) * 0.3 + Math.sin(u * TAU * (f * 5 + 2) + seed * 0.7) * 0.2;
  // Y runs upward in texture space: row 0 is the foot of the picture.
  const texture = createTexture(W, H, 1, (x, y, o) => {
    const u = x / W;
    const h = y / H;
    const far = 0.58 + ridge(u, 3, 1.3) * 0.2;
    const mid = 0.38 + ridge(u, 5, 4.1) * 0.13;
    const trees = 0.2 + ridge(u, 9, 2.2) * 0.05 + Math.abs(Math.sin(u * TAU * 96)) * 0.04 + Math.abs(Math.sin(u * TAU * 211)) * 0.02;
    // Resort blocks on one stretch of the skyline.
    const inResort = u > 0.3 && u < 0.37;
    const tier = 0.3 + Math.floor((Math.abs(u - 0.335) / 0.035) * 3) * -0.05 + 0.1;
    const isResort = inResort && h < tier;
    const grain = random() * 7;
    if (isResort) {
      const windows = y % 4 < 1 ? 0.78 : 1;
      o[0] = 236 * windows;
      o[1] = 228 * windows;
      o[2] = 214 * windows;
    } else if (h < trees) {
      o[0] = 62 + grain;
      o[1] = 96 + grain;
      o[2] = 66 + grain;
    } else if (h < mid) {
      o[0] = 118 + grain;
      o[1] = 150 + grain;
      o[2] = 136 + grain;
    } else if (h < far) {
      o[0] = 158 + grain * 0.5;
      o[1] = 184 + grain * 0.5;
      o[2] = 198 + grain * 0.5;
    } else {
      o[3] = 0; // sky shows through
    }
  });
  texture.wrapT = THREE.ClampToEdgeWrapping; // the top row must not wrap onto the foot
  return texture;
}

function buildBackdropGeometry() {
  const b = new MeshBuilder();
  const n = 20;
  const R = SKY_RADIUS * 0.92;
  const low = -3;
  const high = 54;
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * Math.PI * 2;
    const a1 = ((i + 1) / n) * Math.PI * 2;
    const p = (a: number, y: number) => [Math.cos(a) * R, y, Math.sin(a) * R];
    b.quad(p(a0, low), p(a1, low), p(a1, high), p(a0, high), 0xffffff, [i / n, 0, (i + 1) / n, 0, (i + 1) / n, 1, i / n, 1]);
  }
  return b.build(true);
}

function buildCloudGeometry() {
  const b = new MeshBuilder();
  const random = seededRandom(17);
  for (let i = 0; i < 10; i++) {
    const a = random() * Math.PI * 2;
    const dist = SKY_RADIUS * (0.66 + random() * 0.2);
    const y = 48 + random() * 62;
    const halfW = 34 + random() * 40;
    const halfH = 11 + random() * 9;
    const cx = Math.cos(a) * dist;
    const cz = Math.sin(a) * dist;
    // Tangent to the circle, facing the centre where the camera is.
    const rx = Math.sin(a) * halfW;
    const rz = -Math.cos(a) * halfW;
    b.quad(
      [cx - rx, y - halfH, cz - rz], [cx + rx, y - halfH, cz + rz],
      [cx + rx, y + halfH, cz + rz], [cx - rx, y + halfH, cz - rz],
      0xffffff, [0, 0, 1, 0, 1, 1, 0, 1],
    );
  }
  return b.build(true);
}

function buildTerrainGeometry() {
  const b = new MeshBuilder();
  const E = 1100;
  const tile = 16;
  b.quad(
    [-E, 0, E], [E, 0, E], [E, 0, -E], [-E, 0, -E], 0x7f9f4c,
    [-E / tile, E / tile, E / tile, E / tile, E / tile, -E / tile, -E / tile, -E / tile],
  );
  const seg = TRACK.segment;
  for (let i = 0; i < TRACK.n; i++) {
    const s0 = i * seg;
    const s1 = (i + 1) * seg;
    const k = TRACK.k[i];
    // Gravel run-off on the outside of the faster corners.
    if (Math.abs(k) > 1 / 72) {
      const lo = k > 0 ? -WALL : KERB_OUTER;
      const hi = k > 0 ? -KERB_OUTER : WALL;
      const a = trackVertex(s0, lo, 0);
      const q = trackVertex(s0, hi, 0);
      const c = trackVertex(s1, hi, 0);
      const r = trackVertex(s1, lo, 0);
      const uv = (p: number[]) => [p[0] / tile, p[2] / tile];
      b.quad(a, q, c, r, 0xc9b184, [...uv(a), ...uv(q), ...uv(c), ...uv(r)]);
    }
  }
  return b.build(true);
}

function buildHillGeometry() {
  const b = new MeshBuilder();
  const seg = TRACK.segment;
  const tile = 12;
  const fractions = [0, 1];
  const shade = (s: number, side: number, row: number) => {
    const a = (s / TRACK.length) * Math.PI * 2;
    const dryness = clamp(0.5 + Math.sin(a * 5 + side) * 0.32 + Math.sin(a * 11 - side * 2) * 0.2 + row * 0.08, 0, 1);
    return mixColor(0x5f8f3a, 0xc2ad62, dryness);
  };
  for (let i = 0; i < TRACK.n; i++) {
    const s0 = i * seg;
    const s1 = (i + 1) * seg;
    for (const side of [-1, 1]) {
      for (let row = 0; row < fractions.length - 1; row++) {
        const inner = row;
        const outer = row + 1;
        // Lower-offset edge first so every quad winds upward on both sides.
        const lo = side < 0 ? outer : inner;
        const hi = side < 0 ? inner : outer;
        const vertex = (sampleIndex: number, s: number, f: number) => {
          const d = hillOffset(sampleIndex, side, fractions[f]);
          return trackVertex(s, d, hillHeight(s, d));
        };
        const a = vertex(i, s0, lo);
        const q = vertex(i, s0, hi);
        const c = vertex(i + 1, s1, hi);
        const r = vertex(i + 1, s1, lo);
        const uv = (p: number[]) => [p[0] / tile, p[2] / tile];
        b.quad(a, q, c, r, [shade(s0, side, lo), shade(s0, side, hi), shade(s1, side, hi), shade(s1, side, lo)], [
          ...uv(a), ...uv(q), ...uv(c), ...uv(r),
        ]);
      }
    }
  }
  return b.build(true);
}

function buildRoadGeometry() {
  const b = new MeshBuilder();
  const seg = TRACK.segment;
  const tiles = Math.max(1, Math.round(TRACK.length / ROAD_TILE_METRES));
  for (let i = 0; i < TRACK.n; i++) {
    const s0 = i * seg;
    const s1 = (i + 1) * seg;
    const v0 = (i / TRACK.n) * tiles;
    const v1 = ((i + 1) / TRACK.n) * tiles;
    b.quad(
      trackVertex(s0, -KERB_OUTER, 0), trackVertex(s0, KERB_OUTER, 0),
      trackVertex(s1, KERB_OUTER, 0), trackVertex(s1, -KERB_OUTER, 0),
      0xffffff, [0, v0, 1, v0, 1, v1, 0, v1],
    );
  }
  return b.build(true);
}

/** Red and white kerbs only where a real circuit has them: through the corners. */
function buildKerbGeometry() {
  const b = new MeshBuilder();
  const seg = TRACK.segment;
  for (let i = 0; i < TRACK.n; i++) {
    if (Math.abs(TRACK.k[i]) < 1 / 95) continue;
    const s0 = i * seg;
    const s1 = (i + 1) * seg;
    const u0 = s0 / 6;
    const u1 = s1 / 6;
    for (const [lo, hi] of [[ROAD_HALF, KERB_OUTER], [-KERB_OUTER, -ROAD_HALF]]) {
      b.quad(
        trackVertex(s0, lo, 0), trackVertex(s0, hi, 0),
        trackVertex(s1, hi, 0), trackVertex(s1, lo, 0),
        0xffffff, [u0, 0, u0, 1, u1, 1, u1, 0],
      );
    }
  }
  return b.build(true);
}

function buildDecalGeometry() {
  const b = new MeshBuilder();
  // Chequered start line.
  const cell = (2 * ROAD_HALF) / 10;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 10; col++) {
      const d0 = -ROAD_HALF + col * cell;
      const s0 = row * 1.2;
      b.quad(
        trackVertex(s0, d0, 0), trackVertex(s0, d0 + cell, 0),
        trackVertex(s0 + 1.2, d0 + cell, 0), trackVertex(s0 + 1.2, d0, 0),
        (row + col) % 2 ? 0x16171a : 0xf0f0f0,
      );
    }
  }
  // Boost pads: an orange plate with one chevron.
  for (let i = 0; i < PAD_FRACTIONS.length; i++) {
    const s = PAD_FRACTIONS[i] * TRACK.length;
    const d = PAD_LATERAL[i];
    b.quad(
      trackVertex(s - 3.5, d - 2.5, 0), trackVertex(s - 3.5, d + 2.5, 0),
      trackVertex(s + 3.5, d + 2.5, 0), trackVertex(s + 3.5, d - 2.5, 0),
      0xff7a1a,
    );
    b.quad(
      trackVertex(s - 2, d - 2, 0), trackVertex(s - 0.4, d, 0),
      trackVertex(s + 2.2, d, 0), trackVertex(s + 0.6, d - 2, 0),
      0xffe14d,
    );
    b.quad(
      trackVertex(s - 0.4, d, 0), trackVertex(s - 2, d + 2, 0),
      trackVertex(s + 0.6, d + 2, 0), trackVertex(s + 2.2, d, 0),
      0xffe14d,
    );
  }
  return b.build();
}

const PAD_LATERAL = [-3.5, 3.5, 0, -3];

function buildBarrierGeometry() {
  const b = new MeshBuilder();
  const seg = TRACK.segment;
  const H = 1.25;
  const repeats = Math.max(1, Math.round(TRACK.length / 8));
  for (let i = 0; i < TRACK.n; i++) {
    const s0 = i * seg;
    const s1 = (i + 1) * seg;
    const u0 = (i / TRACK.n) * repeats;
    const u1 = ((i + 1) / TRACK.n) * repeats;
    // Each wall winds so its front face looks at the road.
    b.quad(
      trackVertex(s1, WALL, 0), trackVertex(s0, WALL, 0),
      trackVertex(s0, WALL, H), trackVertex(s1, WALL, H),
      0xffffff, [u1, 0, u0, 0, u0, 1, u1, 1],
    );
    b.quad(
      trackVertex(s0, -WALL, 0), trackVertex(s1, -WALL, 0),
      trackVertex(s1, -WALL, H), trackVertex(s0, -WALL, H),
      0xffffff, [u0, 0, u1, 0, u1, 1, u0, 1],
    );
  }
  return b.build(true);
}

/** Corner marker boards on the outside of each bend, facing oncoming traffic. */
function buildChevronGeometry() {
  const b = new MeshBuilder();
  const seg = TRACK.segment;
  let cooldown = 0;
  for (let i = 0; i < TRACK.n; i++) {
    cooldown--;
    const k = TRACK.k[i];
    if (Math.abs(k) < 1 / 70 || cooldown > 0) continue;
    cooldown = 3;
    const bendsRight = k > 0;
    // Far enough out that the whole board clears the wall and every kart beside it.
    const d = bendsRight ? -(WALL + 4.2) : WALL + 4.2;
    const s = i * seg;
    for (const offset of [-1.7, 1.7]) {
      const y0 = hillHeight(s, d + offset) + 0.9;
      const y1 = y0 + 2.3;
      const left = trackVertex(s, d + offset - 1.55, y0);
      const right = trackVertex(s, d + offset + 1.55, y0);
      const leftTop = trackVertex(s, d + offset - 1.55, y1);
      const rightTop = trackVertex(s, d + offset + 1.55, y1);
      const uv = bendsRight ? [0, 0, 1, 0, 1, 1, 0, 1] : [1, 0, 0, 0, 0, 1, 1, 1];
      b.quad(left, right, rightTop, leftTop, 0xffffff, uv);
    }
  }
  return b.build(true);
}

/** Start gantry: ordinary depth-tested geometry. */
function buildSceneryGeometry() {
  const b = new MeshBuilder();
  const place = (s: number, d: number, y: number, w: number, h: number, depth: number, color: number) => {
    const [x, , z] = trackVertex(s, d, 0);
    b.add(new THREE.BoxGeometry(w, h, depth), color, transform(x, y, z, 0, Math.atan2(trackTangent(s)[0], trackTangent(s)[1])));
  };

  place(0, -WALL, 3.6, 0.6, 7.2, 0.6, 0x2a2c30);
  place(0, WALL, 3.6, 0.6, 7.2, 0.6, 0x2a2c30);
  place(0, 0, 7.4, 2 * WALL + 0.8, 1.7, 0.45, 0x17191d);
  place(-0.3, 0, 7.4, 2 * WALL - 2, 1.05, 0.1, 0xffdc4a);

  return b.build();
}

function trackTangent(s: number) {
  const p: TrackPoint = { x: 0, z: 0, tx: 0, tz: 1, k: 0 };
  sampleTrack(s, p);
  return [p.tx, p.tz];
}

function buildPalmGeometry() {
  const b = new MeshBuilder();
  // A slightly leaning trunk in four stacked sections.
  const sections = 4;
  const height = 8.5;
  let topX = 0;
  for (let i = 0; i < sections; i++) {
    const y = (i + 0.5) * (height / sections);
    const lean = 0.05 + i * 0.05;
    const x = topX + Math.sin(lean) * (height / sections) * 0.5;
    b.add(
      new THREE.CylinderGeometry(0.2 - i * 0.025, 0.28 - i * 0.025, height / sections + 0.05, 5, 1, true),
      i % 2 ? 0x7a6143 : 0x6d563a,
      transform(x, y, 0, 0, 0, -lean),
    );
    topX += Math.sin(lean) * (height / sections);
  }
  // Arching fronds, three tapering quads each.
  const fronds = 9;
  const radii = [0, 1.7, 3.2, 4.3];
  const drops = [0, 0.55, 0.05, -1.5];
  const widths = [0.12, 0.62, 0.42, 0.04];
  for (let f = 0; f < fronds; f++) {
    const a = (f / fronds) * Math.PI * 2;
    const dirX = Math.cos(a);
    const dirZ = Math.sin(a);
    const sideX = -dirZ;
    const sideZ = dirX;
    const lift = f % 2 ? 0.35 : 0;
    const green = f % 3 === 0 ? 0x4f8f35 : f % 3 === 1 ? 0x3d7a2c : 0x62a03e;
    for (let n = 0; n < radii.length - 1; n++) {
      const point = (i: number, sign: number) => [
        topX + dirX * radii[i] + sideX * widths[i] * sign,
        height + drops[i] + lift,
        dirZ * radii[i] + sideZ * widths[i] * sign,
      ];
      b.quad(point(n, -1), point(n + 1, -1), point(n + 1, 1), point(n, 1), green);
    }
  }
  return b.build();
}

/** A lumpy shrub: a subdivided ball with its surface pushed in and out irregularly. */
function buildBushGeometry() {
  const ball = new THREE.IcosahedronGeometry(1, 1);
  const p = ball.getAttribute('position');
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    // Derived from position so that corners shared between faces stay joined.
    const lump = 0.72 + 0.42 * Math.abs(Math.sin(x * 7.3 + y * 3.1) * Math.cos(z * 6.7 - y * 2.3));
    p.setXYZ(i, x * lump, y * lump * 0.85, z * lump);
  }
  return new MeshBuilder().add(ball, 0xffffff).build();
}

function buildShadowGeometry() {
  const plane = new THREE.PlaneGeometry(1, 1);
  plane.rotateX(-Math.PI / 2);
  return plane;
}

/* -------------------------------------------------------------------------- */
/* The car: a lofted sports-car body with glossy matcap paint. Nose faces +Z. */
/* -------------------------------------------------------------------------- */

/**
 * A matcap is a picture of a shiny ball: each pixel is looked up from the surface
 * normal alone. It gives clear-coat reflections for the price of one texture read,
 * far cheaper than real specular lighting on a Mali G52.
 */
function createPaintMatcap(paint: number, reflection = 1) {
  const S = 128;
  const [pr, pg, pb] = rgb(paint);
  const smooth = (a: number, b: number, v: number) => {
    const t = clamp((v - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  };
  return createTexture(S, S, 1, (x, y, o) => {
    const nx = ((x + 0.5) / S) * 2 - 1;
    const ny = ((y + 0.5) / S) * 2 - 1;
    const r = Math.min(1, Math.hypot(nx, ny));
    const body = 0.36 + 0.66 * smooth(-0.8, 0.8, ny);
    // Crisp horizon: bright sky above, dark road below, as on real clear coat.
    const sky = smooth(-0.03, 0.05, ny);
    const fresnel = (0.04 + 0.5 * r * r * r * r) * reflection;
    const amount = fresnel * (0.3 + 0.45 * sky);
    const reflectR = sky * 0.76 * (0.72 + 0.28 * ny) + (1 - sky) * 0.09;
    const reflectG = sky * 0.87 * (0.72 + 0.28 * ny) + (1 - sky) * 0.09;
    const reflectB = sky * 1 * (0.72 + 0.28 * ny) + (1 - sky) * 0.1;
    // Long soft-box highlight along the top of the panels, plus a small sun glint.
    const boxX = (nx + 0.2) / 0.34;
    const boxY = (ny - 0.72) / 0.05;
    const glintX = (nx - 0.42) / 0.11;
    const glintY = (ny - 0.32) / 0.11;
    const highlight =
      0.42 * reflection * Math.exp(-(boxX * boxX + boxY * boxY)) +
      0.75 * reflection * Math.exp(-(glintX * glintX + glintY * glintY));
    o[0] = (pr * body * (1 - amount) + reflectR * amount + highlight) * 255;
    o[1] = (pg * body * (1 - amount) + reflectG * amount + highlight) * 255;
    o[2] = (pb * body * (1 - amount) + reflectB * amount + highlight) * 255;
  });
}

const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function decodeBase64(text: string) {
  const lookup = new Uint8Array(128);
  for (let i = 0; i < BASE64.length; i++) lookup[BASE64.charCodeAt(i)] = i;
  const padding = text.endsWith('==') ? 2 : text.endsWith('=') ? 1 : 0;
  const bytes = new Uint8Array((text.length * 3) / 4 - padding);
  let out = 0;
  for (let i = 0; i < text.length; i += 4) {
    const n =
      (lookup[text.charCodeAt(i)] << 18) |
      (lookup[text.charCodeAt(i + 1)] << 12) |
      (lookup[text.charCodeAt(i + 2)] << 6) |
      lookup[text.charCodeAt(i + 3)];
    if (out < bytes.length) bytes[out++] = (n >> 16) & 255;
    if (out < bytes.length) bytes[out++] = (n >> 8) & 255;
    if (out < bytes.length) bytes[out++] = n & 255;
  }
  return bytes.buffer;
}

function decodeMesh(mesh: EncodedMesh) {
  const positions = new Int16Array(decodeBase64(mesh.p));
  const normals = new Int8Array(decodeBase64(mesh.n));
  const colors = new Uint8Array(decodeBase64(mesh.c));
  const position = new Float32Array(positions.length);
  for (let i = 0; i < position.length; i++) position[i] = (positions[i] / 32767) * CAR_POSITION_SCALE;
  const normal = new Float32Array(normals.length);
  for (let i = 0; i < normal.length; i++) normal[i] = normals[i] / 127;
  const color = new Float32Array(colors.length);
  for (let i = 0; i < color.length; i++) color[i] = colors[i] / 255;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(position, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  g.setAttribute('color', new THREE.BufferAttribute(color, 3));
  g.setIndex(new THREE.BufferAttribute(new Uint16Array(decodeBase64(mesh.i)), 1));
  return g;
}

function indexedTriangles(g: THREE.BufferGeometry) {
  return g.index ? g.index.count / 3 : triangleCount(g);
}

function buildItemBoxGeometry() {
  const b = new MeshBuilder();
  const S = 1.7;
  const face = S / 2 + 0.04;
  b.add(new THREE.BoxGeometry(S, S, S), 0xe0a800);
  // A blocky '?' of five planes, on the front and back faces.
  const bars: [number, number, number, number][] = [
    [0, 0.5, 0.74, 0.18], // top
    [0.37, 0.28, 0.18, 0.44], // right
    [0.12, 0.05, 0.5, 0.18], // middle
    [-0.05, -0.14, 0.18, 0.28], // stem
    [-0.05, -0.5, 0.18, 0.18], // dot
  ];
  for (const ry of [0, Math.PI]) {
    for (const [x, y, w, h] of bars) {
      const offset = new THREE.Matrix4().makeRotationY(ry).multiply(transform(x, y, face));
      b.add(new THREE.PlaneGeometry(w, h), 0x4a2600, offset);
    }
  }
  return b.build();
}

function buildBananaGeometry() {
  return new MeshBuilder()
    .add(new THREE.TorusGeometry(0.55, 0.2, 3, 5, Math.PI), 0xf7d71e, transform(0, 0.25, 0, 0, 0, Math.PI))
    .build();
}

function buildShellGeometry() {
  return new MeshBuilder()
    .add(new THREE.SphereGeometry(0.62, 6, 3), 0x23a53a, transform(0, 0.42, 0, 0, 0, 0, 1, 0.72, 1))
    .build();
}

/* -------------------------------------------------------------------------- */
/* Game state                                                                 */
/* -------------------------------------------------------------------------- */

/** Reusable math objects. The frame loop never allocates a Vector3 or Matrix4. */
interface Scratch {
  position: THREE.Vector3;
  scale: THREE.Vector3;
  cameraTarget: THREE.Vector3;
  cameraLook: THREE.Vector3;
  euler: THREE.Euler;
  quaternion: THREE.Quaternion;
  matrix: THREE.Matrix4;
  point: TrackPoint;
  aheadPoint: TrackPoint;
}

function createScratch(): Scratch {
  return {
    position: new THREE.Vector3(),
    scale: new THREE.Vector3(1, 1, 1),
    cameraTarget: new THREE.Vector3(),
    cameraLook: new THREE.Vector3(),
    euler: new THREE.Euler(),
    quaternion: new THREE.Quaternion(),
    matrix: new THREE.Matrix4(),
    point: { x: 0, z: 0, tx: 0, tz: 1, k: 0 },
    aheadPoint: { x: 0, z: 0, tx: 0, tz: 1, k: 0 },
  };
}

interface SensorState {
  hasGyro: boolean;
  hasAccel: boolean;
  rawGyroX: number;
  rawGyroZ: number;
  rawUpX: number;
  rawUpY: number;
  rawUpZ: number;
  gyroRate: number;
  accelRoll: number;
  upX: number;
  upY: number;
  upZ: number;
  tilt: number;
  touchLeft: boolean;
  touchRight: boolean;
  remoteActive: boolean;
  remoteSteer: number;
}

interface Kart {
  mesh: THREE.Group;
  name: string;
  color: string;
  wheelAngle: number;
  isPlayer: boolean;
  gridSlot: number;
  top: number;
  waypoints: Float32Array;
  s: number;
  d: number;
  v: number;
  steer: number;
  spinT: number;
  boostT: number;
  boostMul: number;
  knockV: number;
  place: number; // 0 until the kart finishes
  x: number;
  z: number;
}

interface WorldItem {
  active: boolean;
  s: number;
  d: number;
  t: number;
}

interface StandingRow {
  name: string;
  color: string;
  isPlayer: boolean;
}

interface HudCallbacks {
  setRank: (v: number) => void;
  setLap: (v: number) => void;
  setSpeed: (v: number) => void;
  setItem: (v: ItemKind | null) => void;
  setPhase: (v: Phase) => void;
  setCountdown: (v: number) => void;
  setFinishTime: (v: number) => void;
  setProgress: (v: number) => void;
  setClock: (v: number) => void;
  setStandings: (v: StandingRow[]) => void;
}

interface Game {
  scratch: Scratch;
  sensors: SensorState;
  hud: HudCallbacks;
  camera: THREE.PerspectiveCamera;
  sun: THREE.DirectionalLight;
  sunFollows: boolean;
  sky: THREE.Mesh;
  backdrop: THREE.Mesh;
  clouds: THREE.Mesh;
  shadowMesh: THREE.InstancedMesh;
  wheelMesh: THREE.InstancedMesh;
  karts: Kart[];
  player: Kart;
  order: Kart[];
  phase: Phase;
  countdown: number;
  raceTime: number;
  finishCount: number;
  hudTimer: number;
  shownRank: number;
  shownLap: number;
  shownSpeed: number;
  shownCountdown: number;
  shownProgress: number;
  shownClock: number;
  clock: number;
  item: ItemKind | null;
  boxS: Float32Array;
  boxD: Float32Array;
  boxX: Float32Array;
  boxZ: Float32Array;
  boxRespawn: Float32Array;
  boxMesh: THREE.InstancedMesh;
  boxMaterial: THREE.MeshLambertMaterial;
  bananas: WorldItem[];
  bananaMesh: THREE.InstancedMesh;
  nextBanana: number;
  shells: WorldItem[];
  shellMesh: THREE.InstancedMesh;
  particlePositions: Float32Array;
  particleColors: Float32Array;
  particleVelocity: Float32Array;
  particleLife: Float32Array;
  particleKind: Uint8Array;
  particleGeometry: THREE.BufferGeometry;
  particleCursor: number;
  particlesAwake: boolean;
}

function buildWaypoints(bias: number, seed: number) {
  const random = seededRandom(seed);
  const waypoints = new Float32Array(WAYPOINT_COUNT);
  const p: TrackPoint = { x: 0, z: 0, tx: 0, tz: 1, k: 0 };
  for (let i = 0; i < WAYPOINT_COUNT; i++) {
    sampleTrack((i / WAYPOINT_COUNT) * TRACK.length + 8, p);
    const inside = clamp(p.k * 180, -4.5, 4.5); // hug the inside of each corner
    waypoints[i] = clamp(inside + bias + (random() - 0.5) * 1.6, -6, 6);
  }
  return waypoints;
}

function createGame(
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  scratch: Scratch,
  sensors: SensorState,
  hud: HudCallbacks,
  anisotropy: number,
  tier: QualitySettings,
  renderer: THREE.WebGLRenderer,
  sun: THREE.DirectionalLight,
): Game {
  const road = createRoadTexture(anisotropy);
  const groundTexture = createGroundTexture(anisotropy);
  const blob = createBlobTexture();

  // Materials: Lambert everywhere that is lit, unlit only for the sky.
  const lit = new THREE.MeshLambertMaterial({ vertexColors: true });
  const boxMaterial = new THREE.MeshLambertMaterial({
    vertexColors: true,
    emissive: new THREE.Color(0xffb000),
    emissiveIntensity: 0.2,
  });
  // Ground layers draw back to front with depth off: painter's order replaces coplanar
  // depth offsets, so nothing z-fights, and the largest surfaces skip the depth cost.
  const paint = (map: THREE.Texture | null) =>
    new THREE.MeshLambertMaterial({ vertexColors: true, map, depthTest: false, depthWrite: false });

  const addStatic = (geometry: THREE.BufferGeometry, material: THREE.Material, order = 0) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    mesh.receiveShadow = tier.shadowSize > 0 && !(material instanceof THREE.MeshBasicMaterial);
    scene.add(mesh);
    staticTriangles += triangleCount(geometry);
    return mesh;
  };
  let staticTriangles = 0;

  const sky = addStatic(
    buildSkyGeometry(),
    new THREE.MeshBasicMaterial({ vertexColors: true, depthTest: false, depthWrite: false, fog: false }),
    -7,
  );
  const backdrop = addStatic(
    buildBackdropGeometry(),
    new THREE.MeshBasicMaterial({
      map: createBackdropTexture(),
      alphaTest: 0.5,
      depthTest: false,
      depthWrite: false,
      fog: false,
      side: THREE.DoubleSide,
    }),
    -6.5,
  );
  const clouds = addStatic(
    tier.clouds ? buildCloudGeometry() : new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([], 3)),
    new THREE.MeshBasicMaterial({
      map: createCloudTexture(),
      transparent: true,
      depthWrite: false,
      fog: false,
      side: THREE.DoubleSide,
    }),
    0,
  );
  addStatic(buildTerrainGeometry(), paint(groundTexture), -6);
  addStatic(buildRoadGeometry(), paint(road), -5);
  addStatic(buildKerbGeometry(), paint(createKerbTexture()), -4);
  addStatic(buildDecalGeometry(), paint(null), -3);
  addStatic(buildHillGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, map: groundTexture }));
  addStatic(buildBarrierGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, map: createBarrierTexture() }));
  addStatic(buildChevronGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, map: createChevronTexture() }));
  addStatic(buildSceneryGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));

  // Vegetation: one draw call per plant type however many are planted.
  const planting = seededRandom(41);
  const palmMaterial = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const palmGeometry = buildPalmGeometry();
  const palmMesh = new THREE.InstancedMesh(palmGeometry, palmMaterial, Math.max(1, tier.palms));
  palmMesh.frustumCulled = false;
  palmMesh.count = tier.palms;
  palmMesh.castShadow = tier.shadowSize > 0;
  const shadowMesh = new THREE.InstancedMesh(
    buildShadowGeometry(),
    new THREE.MeshBasicMaterial({ map: blob, color: 0x000000, transparent: true, opacity: 0.5, depthWrite: false, fog: false }),
    KART_SHADOWS + tier.palms,
  );
  shadowMesh.frustumCulled = false;
  shadowMesh.renderOrder = 1;
  shadowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  for (let i = 0; i < tier.palms; i++) {
    const s = ((i + planting() * 0.8) / tier.palms) * TRACK.length;
    const side = planting() < 0.5 ? -1 : 1;
    const sample = Math.floor(s / TRACK.segment);
    const d = hillOffset(sample, side, 0.12 + planting() * 0.5);
    const [x, , z] = trackVertex(s, d, 0);
    const y = hillHeight(s, d) - 0.2;
    const size = 0.8 + planting() * 0.55;
    scratch.scale.set(size, size, size);
    scratch.position.set(x, y, z);
    scratch.euler.set(0, planting() * Math.PI * 2, 0);
    scratch.quaternion.setFromEuler(scratch.euler);
    scratch.matrix.compose(scratch.position, scratch.quaternion, scratch.scale);
    palmMesh.setMatrixAt(i, scratch.matrix);
    // A soft shadow pool, offset away from the sun.
    scratch.scale.set(5.5 * size, 1, 5.5 * size);
    scratch.position.set(x - 1.2, y + 0.06, z - 0.8);
    scratch.quaternion.identity();
    scratch.matrix.compose(scratch.position, scratch.quaternion, scratch.scale);
    shadowMesh.setMatrixAt(KART_SHADOWS + i, scratch.matrix);
  }
  scratch.scale.set(1, 1, 1);
  palmMesh.instanceMatrix.needsUpdate = true;
  scene.add(palmMesh);
  scene.add(shadowMesh);

  // Plants beyond the baseline: one instanced draw call however many are planted.
  const bushGeometry = buildBushGeometry();
  const bushMesh = new THREE.InstancedMesh(bushGeometry, lit, Math.max(1, tier.bushes));
  bushMesh.count = tier.bushes;
  bushMesh.frustumCulled = false;
  bushMesh.castShadow = tier.shadowSize > 0;
  const bushTint = new THREE.Color();
  for (let i = 0; i < tier.bushes; i++) {
    const s = planting() * TRACK.length;
    const side = planting() < 0.5 ? -1 : 1;
    const sample = Math.floor(s / TRACK.segment);
    const d = hillOffset(sample, side, 0.05 + planting() * 0.9);
    const [x, , z] = trackVertex(s, d, 0);
    const size = 0.55 + planting() * 0.95;
    scratch.scale.set(size * (0.9 + planting() * 0.5), size * 0.8, size);
    scratch.position.set(x, hillHeight(s, d) + size * 0.35, z);
    scratch.euler.set(0, planting() * 6.28, 0);
    scratch.quaternion.setFromEuler(scratch.euler);
    scratch.matrix.compose(scratch.position, scratch.quaternion, scratch.scale);
    bushMesh.setMatrixAt(i, scratch.matrix);
    bushTint.setHex(mixColor(0x2f6a2a, 0x7c9a3e, planting()));
    bushMesh.setColorAt(i, bushTint);
  }
  scratch.scale.set(1, 1, 1);
  bushMesh.instanceMatrix.needsUpdate = true;
  if (tier.bushes > 0) scene.add(bushMesh);

  // Cars share geometry per level of detail; only the paint differs between them.
  const lodCache: Partial<Record<CarLod, Record<'paint' | 'trim' | 'glass' | 'lights', THREE.BufferGeometry>>> = {};
  const carLod = (lod: CarLod) => {
    let parts = lodCache[lod];
    if (!parts) {
      const m = CAR_MODEL[lod];
      parts = { paint: decodeMesh(m.paint), trim: decodeMesh(m.trim), glass: decodeMesh(m.glass), lights: decodeMesh(m.lights) };
      lodCache[lod] = parts;
    }
    return parts;
  };

  // Real reflections: a small environment captured from the sky, on physically based paint.
  let environment: THREE.Texture | null = null;
  if (tier.realReflections) {
    const envScene = new THREE.Scene();
    envScene.add(new THREE.Mesh(sky.geometry, sky.material));
    const below = new THREE.Mesh(new THREE.CircleGeometry(SKY_RADIUS, 16), new THREE.MeshBasicMaterial({ color: 0x4d4f44, fog: false, side: THREE.DoubleSide }));
    below.rotation.x = -Math.PI / 2;
    below.position.y = -0.5;
    envScene.add(below);
    const generator = new THREE.PMREMGenerator(renderer);
    environment = generator.fromScene(envScene, 0, 1, SKY_RADIUS * 2).texture;
    generator.dispose();
  }
  const paintMaterial = (color: number): THREE.Material =>
    environment
      ? new THREE.MeshPhysicalMaterial({ color, metalness: 0.35, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.04, envMap: environment, envMapIntensity: 1.25 })
      : new THREE.MeshMatcapMaterial({ matcap: createPaintMatcap(color) });
  const trimMaterial: THREE.Material = environment
    ? new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.5, roughness: 0.42, envMap: environment })
    : new THREE.MeshMatcapMaterial({ vertexColors: true, matcap: createPaintMatcap(0xc4c7cd) });
  const glassMaterial: THREE.Material = environment
    ? new THREE.MeshStandardMaterial({ color: 0x07090c, metalness: 0.9, roughness: 0.06, envMap: environment, envMapIntensity: 1.5 })
    : new THREE.MeshMatcapMaterial({ matcap: createPaintMatcap(0x0c141c, 1.5) });
  const lightMaterial = new THREE.MeshBasicMaterial({ vertexColors: true });
  const wheelGeometry = decodeMesh(tier.wheelHd ? CAR_WHEEL_HD_MESH : CAR_WHEEL_MESH);
  const wheelMesh = new THREE.InstancedMesh(
    wheelGeometry,
    trimMaterial,
    KART_SHADOWS * 4,
  );
  wheelMesh.frustumCulled = false;
  wheelMesh.castShadow = tier.shadowSize > 0;
  wheelMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(wheelMesh);

  const kartSpecs: [number, number, boolean, number, number][] = [
    // body, accent, isPlayer, top speed, racing-line bias
    [0xd81818, 0x17181b, true, PLAYER_TOP_SPEED, 0],
    [0x1f5fd6, 0x17181b, false, CPU_TOP_SPEEDS[0], -1.6],
    [0xf0c419, 0x17181b, false, CPU_TOP_SPEEDS[1], 1.6],
    [0x1f9e49, 0x17181b, false, CPU_TOP_SPEEDS[2], 0],
  ];
  let kartTriangles = 0;
  const karts: Kart[] = kartSpecs.map(([body, accent, isPlayer, top, bias], i) => {
    void accent;
    // Geometry is shared by all four cars; only the paint differs.
    const mesh = new THREE.Group();
    const parts = carLod(isPlayer ? tier.playerLod : tier.cpuLod);
    const bodyMesh = new THREE.Mesh(parts.paint, paintMaterial(body));
    bodyMesh.castShadow = tier.shadowSize > 0;
    mesh.add(bodyMesh);
    mesh.add(new THREE.Mesh(parts.trim, trimMaterial));
    mesh.add(new THREE.Mesh(parts.glass, glassMaterial));
    mesh.add(new THREE.Mesh(parts.lights, lightMaterial));
    mesh.matrixAutoUpdate = false; // updated explicitly so the wheels can follow it
    kartTriangles +=
      indexedTriangles(parts.paint) +
      indexedTriangles(parts.trim) +
      indexedTriangles(parts.glass) +
      indexedTriangles(parts.lights) +
      indexedTriangles(wheelGeometry) * 4;
    scene.add(mesh);
    return {
      mesh,
      name: ['YOU', 'BLUE', 'YELLOW', 'GREEN'][i],
      color: `#${body.toString(16).padStart(6, '0')}`,
      isPlayer,
      gridSlot: isPlayer ? 3 : i - 1, // the player starts at the back of the grid
      top,
      waypoints: buildWaypoints(bias, 100 + i),
      s: 0, d: 0, v: 0, steer: 0, spinT: 0, boostT: 0, boostMul: 1, knockV: 0, place: 0, x: 0, z: 0, wheelAngle: 0,
    };
  });

  // Item boxes sit in rows across the road.
  const boxLayout: [number, number][] = [
    [0.25, -4], [0.25, 0], [0.25, 4],
    [0.55, -4], [0.55, 0], [0.55, 4],
    [0.8, -3], [0.8, 3],
  ];
  const boxS = new Float32Array(BOX_COUNT);
  const boxD = new Float32Array(BOX_COUNT);
  const boxX = new Float32Array(BOX_COUNT);
  const boxZ = new Float32Array(BOX_COUNT);
  boxLayout.forEach(([fraction, d], i) => {
    boxS[i] = fraction * TRACK.length;
    boxD[i] = d;
    const [x, , z] = trackVertex(boxS[i], d, 0);
    boxX[i] = x;
    boxZ[i] = z;
  });
  const boxGeometry = buildItemBoxGeometry();
  const boxMesh = new THREE.InstancedMesh(boxGeometry, boxMaterial, BOX_COUNT);
  boxMesh.frustumCulled = false;
  boxMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(boxMesh);

  const bananaGeometry = buildBananaGeometry();
  const bananaMesh = new THREE.InstancedMesh(bananaGeometry, lit, BANANA_COUNT);
  bananaMesh.frustumCulled = false;
  bananaMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(bananaMesh);

  const shellGeometry = buildShellGeometry();
  const shellMesh = new THREE.InstancedMesh(shellGeometry, lit, SHELL_COUNT);
  shellMesh.frustumCulled = false;
  shellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(shellMesh);

  // Boost exhaust: GL points cost no triangles and only one draw call.
  const particlePositions = new Float32Array(PARTICLE_COUNT * 3);
  const particleColors = new Float32Array(PARTICLE_COUNT * 3);
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(particlePositions, 3).setUsage(THREE.DynamicDrawUsage),
  );
  particleGeometry.setAttribute(
    'color',
    new THREE.BufferAttribute(particleColors, 3).setUsage(THREE.DynamicDrawUsage),
  );
  const particles = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({
      size: 0.55,
      map: blob,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
      fog: false,
    }),
  );
  particles.frustumCulled = false;
  particles.renderOrder = 2;
  scene.add(particles);

  const worstCaseTriangles =
    staticTriangles +
    triangleCount(palmGeometry) * tier.palms +
    triangleCount(bushGeometry) * tier.bushes +
    2 * (KART_SHADOWS + tier.palms) +
    kartTriangles +
    triangleCount(boxGeometry) * BOX_COUNT +
    triangleCount(bananaGeometry) * BANANA_COUNT +
    triangleCount(shellGeometry) * SHELL_COUNT;
  if (__DEV__) {
    console.log(`[kart] worst-case scene triangles: ${worstCaseTriangles}`);
  }

  const makeItems = (count: number): WorldItem[] =>
    Array.from({ length: count }, () => ({ active: false, s: 0, d: 0, t: 0 }));

  const game: Game = {
    scratch,
    sensors,
    hud,
    camera,
    sun,
    sunFollows: tier.shadowSize > 0,
    sky,
    backdrop,
    clouds,
    shadowMesh,
    wheelMesh,
    karts,
    player: karts[0],
    order: karts.slice(),
    phase: 'countdown',
    countdown: 3,
    raceTime: 0,
    finishCount: 0,
    hudTimer: 0,
    shownRank: 4,
    shownLap: 1,
    shownSpeed: 0,
    shownCountdown: 3,
    shownProgress: 0,
    shownClock: 0,
    clock: 0,
    item: null,
    boxS,
    boxD,
    boxX,
    boxZ,
    boxRespawn: new Float32Array(BOX_COUNT),
    boxMesh,
    boxMaterial,
    bananas: makeItems(BANANA_COUNT),
    bananaMesh,
    nextBanana: 0,
    shells: makeItems(SHELL_COUNT),
    shellMesh,
    particlePositions,
    particleColors,
    particleVelocity: new Float32Array(PARTICLE_COUNT * 3),
    particleLife: new Float32Array(PARTICLE_COUNT),
    particleKind: new Uint8Array(PARTICLE_COUNT),
    particleGeometry,
    particleCursor: 0,
    particlesAwake: false,
  };
  resetRace(game);
  return game;
}

function resetRace(g: Game) {
  for (const k of g.karts) {
    k.s = -6 - k.gridSlot * 6;
    k.d = k.gridSlot % 2 ? 3 : -3;
    k.v = 0;
    k.steer = 0;
    k.spinT = 0;
    k.boostT = 0;
    k.boostMul = 1;
    k.knockV = 0;
    k.place = 0;
  }
  g.phase = 'countdown';
  g.countdown = 3;
  g.raceTime = 0;
  g.finishCount = 0;
  g.item = null;
  g.boxRespawn.fill(0);
  for (const b of g.bananas) b.active = false;
  for (const s of g.shells) s.active = false;
  g.particleLife.fill(0);
  g.particlePositions.fill(0);
  g.particleColors.fill(0);
  g.particlesAwake = true;

  g.shownRank = 4;
  g.shownLap = 1;
  g.shownSpeed = 0;
  g.shownCountdown = 3;
  g.hudTimer = 0;
  g.shownProgress = 0;
  g.shownClock = 0;
  g.hud.setProgress(0);
  g.hud.setClock(0);
  g.hud.setItem(null);
  g.hud.setPhase('countdown');
  g.hud.setCountdown(3);
  g.hud.setLap(1);
  g.hud.setRank(4);
  g.hud.setSpeed(0);

  syncKarts(g);
  placeCamera(g, 1);
}

/* -------------------------------------------------------------------------- */
/* Simulation                                                                 */
/* -------------------------------------------------------------------------- */

function readSteering(g: Game, dt: number) {
  return readSensorSteering(g.sensors, dt);
}

function readSensorSteering(n: SensorState, dt: number) {
  // Listeners only store raw readings; smoothing runs here, once per frame. Android may
  // deliver sensors far slower than requested (160 ms on a Galaxy A14 without the high
  // sampling permission), and a per-sample LERP would then lag by seconds. The exponent
  // keeps alpha at exactly SENSOR_LERP per 60 Hz frame whatever the frame rate.
  const alpha = 1 - Math.pow(1 - SENSOR_LERP, dt * 60);
  n.upX += (n.rawUpX - n.upX) * alpha;
  n.upY += (n.rawUpY - n.upY) * alpha;
  n.upZ += (n.rawUpZ - n.upZ) * alpha;
  // Landscape, top of the phone to the left: screen-right is device -Y, so it dips as
  // device +Y rises, and upY measures the roll however far the phone is pitched.
  n.accelRoll = Math.atan2(n.upY, Math.sqrt(n.upX * n.upX + n.upZ * n.upZ));
  // The roll rate is the rotation about the axis perpendicular to gravity and screen-right.
  n.gyroRate += (n.rawGyroX * n.upZ - n.rawGyroZ * n.upX - n.gyroRate) * alpha;

  if (n.hasGyro) {
    // A gyroscope reports rotation rate, not angle, so the smoothed rate is
    // integrated into a tilt. Gravity (or a slow leak) cancels integration drift.
    n.tilt += n.gyroRate * dt;
    if (n.hasAccel) n.tilt += (n.accelRoll - n.tilt) * Math.min(1, 2 * dt);
    else n.tilt -= n.tilt * Math.min(1, 0.35 * dt);
    n.tilt = clamp(n.tilt, -1, 1);
  } else if (n.hasAccel) {
    n.tilt = n.accelRoll;
  }

  const touch = (n.touchRight ? 1 : 0) - (n.touchLeft ? 1 : 0);
  if (touch !== 0) return touch;
  if (n.remoteActive) return n.remoteSteer;
  if (!n.hasGyro && !n.hasAccel) return 0;
  const magnitude = Math.abs(n.tilt);
  if (magnitude < TILT_DEADZONE) return 0;
  const steer = Math.min(1, (magnitude - TILT_DEADZONE) / (TILT_FULL_LOCK - TILT_DEADZONE));
  return n.tilt > 0 ? steer : -steer;
}

function aiSteering(g: Game, k: Kart) {
  const ahead = g.scratch.aheadPoint;
  const lookS = k.s + 16;
  sampleTrack(lookS, ahead);
  let u = lookS % TRACK.length;
  if (u < 0) u += TRACK.length;
  const f = (u / TRACK.length) * WAYPOINT_COUNT;
  const i = Math.floor(f) % WAYPOINT_COUNT;
  const j = (i + 1) % WAYPOINT_COUNT;
  const target = k.waypoints[i] + (k.waypoints[j] - k.waypoints[i]) * (f - Math.floor(f));
  sampleTrack(k.s, ahead);
  // Feed-forward cancels the corner push, then a proportional term chases the waypoint.
  const feedForward = (ahead.k * k.v * k.v * CORNER_PUSH) / STEER_RATE;
  return clamp((target - k.d) * 0.3 + feedForward, -1, 1);
}

function stepKart(g: Game, k: Kart, steerInput: number, dt: number) {
  const p = g.scratch.point;
  sampleTrack(k.s, p);

  if (k.spinT > 0) k.spinT = Math.max(0, k.spinT - dt);
  if (k.boostT > 0) k.boostT = Math.max(0, k.boostT - dt);

  const steer = k.spinT > 0 ? 0 : steerInput;
  k.steer += (steer - k.steer) * Math.min(1, 12 * dt);

  let target = k.top;
  if (!k.isPlayer) {
    // Mild rubber-banding keeps the pack in range without the CPUs feeling scripted.
    target *= 1 + clamp((g.player.s - k.s) / 60, -1, 1) * 0.07;
  }
  if (k.boostT > 0) target *= k.boostMul;
  else if (Math.abs(k.d) > KERB_OUTER) target *= OFFROAD_FACTOR;
  if (k.spinT > 0) target *= SPIN_SPEED_FACTOR;
  k.v += (target - k.v) * Math.min(1, (k.v < target ? 1.1 : 2.4) * dt);

  // Forward motion, with the inside of a corner genuinely shorter than the outside.
  const inside = clamp(1 - p.k * k.d, 0.75, 1.25);
  k.s += ((k.v / inside) + k.knockV) * dt;
  k.wheelAngle = (k.wheelAngle + (k.v / 0.38) * dt) % (Math.PI * 2);
  if (k.knockV !== 0) {
    k.knockV *= Math.exp(-4 * dt);
    if (Math.abs(k.knockV) < 0.5) k.knockV = 0;
  }

  const authority = Math.min(1, k.v / 14);
  k.d += (k.steer * STEER_RATE * authority - p.k * k.v * k.v * CORNER_PUSH) * dt;
  if (Math.abs(k.d) > KART_WALL_LIMIT) {
    k.d = k.d > 0 ? KART_WALL_LIMIT : -KART_WALL_LIMIT;
    k.v *= 1 - Math.min(1, 0.9 * dt); // scraping a barrier costs speed
  }
}

function applySpin(k: Kart, speedFactor: number) {
  k.spinT = SPIN_SECONDS;
  k.boostT = 0;
  k.v *= speedFactor;
}

function fireHeldItem(g: Game) {
  if (g.phase !== 'racing' || !g.item) return;
  const p = g.player;
  if (g.item === 'mushroom') {
    p.boostT = MUSHROOM_SECONDS;
    p.boostMul = MUSHROOM_MULTIPLIER;
    p.v = Math.max(p.v, p.top * MUSHROOM_MULTIPLIER);
  } else if (g.item === 'banana') {
    const b = g.bananas[g.nextBanana];
    g.nextBanana = (g.nextBanana + 1) % BANANA_COUNT;
    b.active = true;
    b.s = p.s - 3.5;
    b.d = p.d;
  } else {
    for (let i = 0; i < SHELL_COUNT; i++) {
      const s = g.shells[i];
      if (!s.active) {
        s.active = true;
        s.s = p.s + 2.5;
        s.d = p.d;
        s.t = SHELL_LIFE_SECONDS;
        break;
      }
    }
  }
  g.item = null;
  g.hud.setItem(null);
}

function updatePickups(g: Game, dt: number) {
  const player = g.player;

  for (let i = 0; i < BOX_COUNT; i++) {
    if (g.boxRespawn[i] > 0) {
      g.boxRespawn[i] -= dt;
      continue;
    }
    if (Math.abs(wrapDelta(player.s, g.boxS[i])) < 3 && Math.abs(player.d - g.boxD[i]) < 2.1) {
      g.boxRespawn[i] = BOX_RESPAWN_SECONDS;
      if (!g.item) {
        g.item = ITEM_KINDS[Math.floor(Math.random() * ITEM_KINDS.length)];
        g.hud.setItem(g.item);
      }
    }
  }

  for (let n = 0; n < g.karts.length; n++) {
    const k = g.karts[n];
    for (let i = 0; i < PAD_FRACTIONS.length; i++) {
      const onPad =
        Math.abs(wrapDelta(k.s, PAD_FRACTIONS[i] * TRACK.length)) < 3.5 &&
        Math.abs(k.d - PAD_LATERAL[i]) < 2.5;
      const hasMushroom = k.boostT > 0 && k.boostMul === MUSHROOM_MULTIPLIER;
      if (onPad && k.spinT <= 0 && !hasMushroom) {
        k.boostT = PAD_SECONDS;
        k.boostMul = PAD_MULTIPLIER;
        k.v = Math.max(k.v, k.top * PAD_MULTIPLIER);
      }
    }
    for (let i = 0; i < BANANA_COUNT; i++) {
      const b = g.bananas[i];
      if (b.active && k.spinT <= 0 && Math.abs(wrapDelta(k.s, b.s)) < 2.6 && Math.abs(k.d - b.d) < 1.6) {
        b.active = false;
        applySpin(k, SPIN_SPEED_FACTOR);
      }
    }
  }

  for (let n = 0; n < SHELL_COUNT; n++) {
    const shell = g.shells[n];
    if (!shell.active) continue;
    shell.t -= dt;
    if (shell.t <= 0) {
      shell.active = false;
      continue;
    }
    shell.s += SHELL_SPEED * dt;
    // A gentle pull towards the nearest CPU ahead makes shells feel fair on a phone.
    let nearest: Kart | null = null;
    let nearestGap = 28;
    for (let i = 0; i < g.karts.length; i++) {
      const k = g.karts[i];
      if (k.isPlayer) continue;
      const gap = wrapDelta(k.s, shell.s);
      if (gap >= -2 && gap < nearestGap) {
        nearestGap = gap;
        nearest = k;
      }
      if (Math.abs(gap) < 3.2 && Math.abs(k.d - shell.d) < 2.1) {
        shell.active = false;
        applySpin(k, 0.3);
        k.knockV = -40; // knocked back along the track
        break;
      }
    }
    if (shell.active && nearest) {
      shell.d += clamp(nearest.d - shell.d, -1, 1) * 9 * dt;
    }
    shell.d = clamp(shell.d, -KART_WALL_LIMIT, KART_WALL_LIMIT);
  }
}

function resolveKartContacts(g: Game, dt: number) {
  const karts = g.karts;
  for (let i = 0; i < karts.length; i++) {
    for (let j = i + 1; j < karts.length; j++) {
      const a = karts[i];
      const b = karts[j];
      const ds = wrapDelta(a.s, b.s);
      const dd = a.d - b.d;
      if (Math.abs(ds) < 4.4 && Math.abs(dd) < 1.9) {
        const push = (1.9 - Math.abs(dd)) * 0.5;
        const side = dd >= 0 ? 1 : -1;
        a.d = clamp(a.d + side * push, -KART_WALL_LIMIT, KART_WALL_LIMIT);
        b.d = clamp(b.d - side * push, -KART_WALL_LIMIT, KART_WALL_LIMIT);
        const rear = ds < 0 ? a : b;
        rear.v *= 1 - Math.min(1, 1.5 * dt);
      }
    }
  }
}

function raceOrderBefore(a: Kart, b: Kart) {
  if (a.place && b.place) return a.place < b.place;
  if (a.place) return true;
  if (b.place) return false;
  return a.s > b.s;
}

/** Insertion sort in place: allocation-free, and nearly sorted from frame to frame. */
function sortRaceOrder(g: Game) {
  const order = g.order;
  for (let i = 1; i < order.length; i++) {
    const kart = order[i];
    let j = i - 1;
    while (j >= 0 && raceOrderBefore(kart, order[j])) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = kart;
  }
}

function endRace(g: Game) {
  g.phase = 'finished';
  sortRaceOrder(g);
  g.hud.setStandings(g.order.map((k) => ({ name: k.name, color: k.color, isPlayer: k.isPlayer })));
  g.item = null;
  g.hud.setItem(null);
  g.hud.setFinishTime(g.raceTime);
  g.hud.setRank(g.player.place);
  g.hud.setLap(LAPS);
  g.hud.setPhase('finished');
}

function updateRace(g: Game, dt: number) {
  g.clock += dt;
  const steerInput = readSteering(g, dt);

  if (g.phase === 'countdown') {
    g.countdown -= dt;
    const shown = Math.ceil(g.countdown);
    if (shown !== g.shownCountdown) g.hud.setCountdown((g.shownCountdown = shown));
    if (g.countdown <= 0) {
      g.phase = 'racing';
      g.hud.setPhase('racing');
    }
    return;
  }

  if (g.phase === 'racing') g.raceTime += dt;

  for (let i = 0; i < g.karts.length; i++) {
    const k = g.karts[i];
    const driverless = !k.isPlayer || g.phase === 'finished';
    stepKart(g, k, driverless ? aiSteering(g, k) : steerInput, dt);
  }
  updatePickups(g, dt);
  resolveKartContacts(g, dt);

  if (g.phase === 'racing') {
    for (let i = 0; i < g.karts.length; i++) {
      const k = g.karts[i];
      if (!k.place && k.s >= LAPS * TRACK.length) k.place = ++g.finishCount;
    }
    if (!g.player.place && g.finishCount === g.karts.length - 1) {
      g.player.place = ++g.finishCount; // every CPU is home: the race is over
    }
    if (g.player.place) endRace(g);
  }
  sortRaceOrder(g);

  g.hudTimer -= dt;
  if (g.hudTimer <= 0 && g.phase === 'racing') {
    g.hudTimer = 0.15;
    // Calling a state setter allocates even when the value is unchanged, so compare first.
    const rank = g.order.indexOf(g.player) + 1;
    const lap = Math.min(LAPS, Math.floor(Math.max(0, g.player.s) / TRACK.length) + 1);
    const speed = Math.round(g.player.v * 3.6);
    if (rank !== g.shownRank) g.hud.setRank((g.shownRank = rank));
    if (lap !== g.shownLap) g.hud.setLap((g.shownLap = lap));
    if (speed !== g.shownSpeed) g.hud.setSpeed((g.shownSpeed = speed));
    const progress = clamp(Math.floor((g.player.s / (LAPS * TRACK.length)) * 100), 0, 100);
    if (progress !== g.shownProgress) g.hud.setProgress((g.shownProgress = progress));
    const tenths = Math.floor(g.raceTime * 10);
    if (tenths !== g.shownClock) g.hud.setClock((g.shownClock = tenths));
  }
}

/* -------------------------------------------------------------------------- */
/* Syncing simulation state to the scene                                      */
/* -------------------------------------------------------------------------- */

function syncKarts(g: Game) {
  const p = g.scratch.point;
  for (let i = 0; i < g.karts.length; i++) {
    const k = g.karts[i];
    sampleTrack(k.s, p);
    k.x = p.x - p.tz * k.d;
    k.z = p.z + p.tx * k.d;
    const spin = k.spinT > 0 ? ((SPIN_SECONDS - k.spinT) / SPIN_SECONDS) * Math.PI * 2 : 0;
    const yaw = Math.atan2(p.tx, p.tz) - k.steer * 0.28 + spin;
    const boosting = k.boostT > 0 ? 1 : 0;
    k.mesh.position.set(k.x, 0, k.z);
    k.mesh.rotation.set(-0.035 * boosting, yaw, -k.steer * 0.05);
    // Soft contact shadow, stretched along the car.
    g.scratch.scale.set(3, 1, 6.1);
    writeInstance(g.shadowMesh, i, g, k.x, 0.03, k.z, yaw);
    g.scratch.scale.set(1, 1, 1);

    // Wheels follow the car, roll with its speed, and the front pair steers.
    k.mesh.updateMatrix();
    const { position, scale, euler, quaternion, matrix } = g.scratch;
    for (let w = 0; w < 4; w++) {
      const wheel = CAR_WHEEL_POSITIONS[w];
      const left = wheel[0] < 0;
      euler.set(left ? -k.wheelAngle : k.wheelAngle, (left ? Math.PI : 0) + (w < 2 ? -k.steer * 0.32 : 0), 0, 'YXZ');
      quaternion.setFromEuler(euler);
      position.set(wheel[0], wheel[2], wheel[1]);
      matrix.compose(position, quaternion, scale);
      matrix.premultiply(k.mesh.matrix);
      g.wheelMesh.setMatrixAt(i * 4 + w, matrix);
    }
    scale.set(1, 1, 1);
  }
  g.wheelMesh.instanceMatrix.needsUpdate = true;
}

function writeInstance(mesh: THREE.InstancedMesh, index: number, g: Game, x: number, y: number, z: number, yaw: number) {
  const { position, scale, euler, quaternion, matrix } = g.scratch;
  position.set(x, y, z);
  euler.set(0, yaw, 0);
  quaternion.setFromEuler(euler);
  matrix.compose(position, quaternion, scale);
  mesh.setMatrixAt(index, matrix);
}

function syncItems(g: Game) {
  const p = g.scratch.point;

  let boxes = 0;
  for (let i = 0; i < BOX_COUNT; i++) {
    if (g.boxRespawn[i] > 0) continue;
    writeInstance(g.boxMesh, boxes++, g, g.boxX[i], 1.7 + Math.sin(g.clock * 2.4 + i) * 0.18, g.boxZ[i], g.clock * 1.3 + i);
  }
  g.boxMesh.count = boxes;
  g.boxMesh.instanceMatrix.needsUpdate = true;
  g.boxMaterial.emissiveIntensity = 0.22 + Math.sin(g.clock * 5) * 0.14;

  let bananas = 0;
  for (let i = 0; i < BANANA_COUNT; i++) {
    const b = g.bananas[i];
    if (!b.active) continue;
    sampleTrack(b.s, p);
    writeInstance(g.bananaMesh, bananas++, g, p.x - p.tz * b.d, 0, p.z + p.tx * b.d, b.d);
  }
  g.bananaMesh.count = bananas;
  g.bananaMesh.instanceMatrix.needsUpdate = true;

  let shells = 0;
  for (let i = 0; i < SHELL_COUNT; i++) {
    const s = g.shells[i];
    if (!s.active) continue;
    sampleTrack(s.s, p);
    writeInstance(g.shellMesh, shells++, g, p.x - p.tz * s.d, 0, p.z + p.tx * s.d, g.clock * 14);
  }
  g.shellMesh.count = shells;
  g.shellMesh.instanceMatrix.needsUpdate = true;
  g.shadowMesh.instanceMatrix.needsUpdate = true;
}

function updateParticles(g: Game, dt: number) {
  const player = g.player;
  const pos = g.particlePositions;
  const col = g.particleColors;
  const vel = g.particleVelocity;
  const life = g.particleLife;
  const p = g.scratch.point;
  let awake = false;

  const skidding = Math.abs(player.steer) > 0.72 && player.v > 22;
  if (g.phase !== 'countdown' && (player.boostT > 0 || skidding)) {
    sampleTrack(player.s, p);
    const flame = player.boostT > 0;
    for (let n = 0; n < (flame ? 2 : 1); n++) {
      const i = g.particleCursor;
      g.particleCursor = (i + 1) % PARTICLE_COUNT;
      life[i] = PARTICLE_LIFE_SECONDS;
      g.particleKind[i] = flame ? 0 : 1;
      const i3 = i * 3;
      const lateral = flame ? (n === 0 ? -0.5 : 0.5) : Math.random() < 0.5 ? -0.95 : 0.95;
      const back = flame ? 2.25 : 1.3;
      pos[i3] = player.x - p.tx * back - p.tz * lateral + (Math.random() - 0.5) * 0.2;
      pos[i3 + 1] = flame ? 0.36 : 0.12;
      pos[i3 + 2] = player.z - p.tz * back + p.tx * lateral + (Math.random() - 0.5) * 0.2;
      // Exhaust keeps most of the car's speed, so it trails behind and never hits the lens.
      vel[i3] = p.tx * player.v * 0.72 + (Math.random() - 0.5) * 2;
      vel[i3 + 1] = 0.6 + Math.random() * 1.8;
      vel[i3 + 2] = p.tz * player.v * 0.72 + (Math.random() - 0.5) * 2;
    }
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const i3 = i * 3;
    if (life[i] <= 0) {
      col[i3] = col[i3 + 1] = col[i3 + 2] = 0; // additive black draws nothing
      continue;
    }
    awake = true;
    life[i] -= dt;
    pos[i3] += vel[i3] * dt;
    pos[i3 + 1] += vel[i3 + 1] * dt;
    pos[i3 + 2] += vel[i3 + 2] * dt;
    const fade = Math.max(0, life[i] / PARTICLE_LIFE_SECONDS);
    if (g.particleKind[i] === 0) {
      col[i3] = fade;
      col[i3 + 1] = 0.62 * fade * fade;
      col[i3 + 2] = 0.1 * fade * fade;
    } else {
      col[i3] = col[i3 + 1] = col[i3 + 2] = 0.3 * fade; // pale tyre smoke
    }
  }

  // Upload only when something changed, plus one frame to flush the last fade-out.
  if (awake || g.particlesAwake) {
    g.particleGeometry.getAttribute('position').needsUpdate = true;
    g.particleGeometry.getAttribute('color').needsUpdate = true;
  }
  g.particlesAwake = awake;
}

function placeCamera(g: Game, follow: number) {
  const { cameraTarget, cameraLook, point: p } = g.scratch;
  const player = g.player;
  sampleTrack(player.s, p);
  // Low, close chase camera with a touch of road vibration at speed.
  const vibration = Math.sin(g.clock * 41) * 0.008 * Math.min(1, player.v / 30);
  cameraTarget.set(player.x - p.tx * 6.2, 2.05 + vibration, player.z - p.tz * 6.2);
  cameraLook.set(player.x + p.tx * 10, 1.05, player.z + p.tz * 10);
  g.camera.position.lerp(cameraTarget, follow);
  g.camera.lookAt(cameraLook);
  g.camera.rotateZ(-player.steer * 0.022); // lean gently into corners

  // Far scenery travels with the camera, which is what makes it read as distant.
  g.sky.position.set(g.camera.position.x, 0, g.camera.position.z);
  g.clouds.position.set(g.camera.position.x, 0, g.camera.position.z);
  g.backdrop.position.set(g.camera.position.x, 0, g.camera.position.z);
  if (g.sunFollows) {
    // The shadow map covers only the area around the player.
    g.sun.position.set(player.x + 44, 60, player.z + 28);
    g.sun.target.position.set(player.x, 0, player.z);
    g.sun.target.updateMatrixWorld();
  }

  const boosting = player.boostT > 0 && player.boostMul === MUSHROOM_MULTIPLIER;
  const fov = boosting ? 70 : 60;
  if (Math.abs(g.camera.fov - fov) > 0.05) {
    g.camera.fov += (fov - g.camera.fov) * Math.min(1, follow * 0.6);
    g.camera.updateProjectionMatrix();
  }
}

/* -------------------------------------------------------------------------- */
/* React                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * expo-gl sizes its drawing buffer from the view's layout at the device pixel ratio,
 * and three's setPixelRatio cannot change that: a smaller viewport would only draw
 * into a corner of the buffer. So the view itself is laid out smaller and scaled up,
 * which is what actually reduces the number of shaded pixels.
 */
const GameSurface = memo(function GameSurface({
  width,
  height,
  bufferScale,
  onContextCreate,
}: {
  width: number;
  height: number;
  bufferScale: number;
  onContextCreate: (gl: ExpoWebGLRenderingContext) => void;
}) {
  const BUFFER_SCALE = bufferScale;
  const bufferWidth = Math.ceil(width * BUFFER_SCALE);
  const bufferHeight = Math.ceil(height * BUFFER_SCALE);
  return (
    <GLView
      msaaSamples={0}
      onContextCreate={onContextCreate}
      style={{
        position: 'absolute',
        left: (width - bufferWidth) / 2,
        top: (height - bufferHeight) / 2,
        width: bufferWidth,
        height: bufferHeight,
        transform: [{ scale: 1 / BUFFER_SCALE }],
      }}
    />
  );
});

type Screen = 'menu' | 'race' | 'controller';

export default function App() {
  useKeepAwake(); // tilt-only play generates no touches, so the screen would time out

  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [rank, setRank] = useState(4);
  const [lap, setLap] = useState(1);
  const [speed, setSpeed] = useState(0);
  const [item, setItem] = useState<ItemKind | null>(null);
  const [phase, setPhase] = useState<Phase>('countdown');
  const [countdown, setCountdown] = useState(3);
  const [finishTime, setFinishTime] = useState(0);
  const [progress, setProgress] = useState(0);
  const [clockTenths, setClock] = useState(0);
  const [standings, setStandings] = useState<StandingRow[]>([]);
  const [gyroAvailable, setGyroAvailable] = useState(true);
  const [screen, setScreen] = useState<Screen>('menu');
  const [quality, setQuality] = useState<Quality>(IS_WEB ? 'ultra' : 'medium');
  const [fps, setFps] = useState(0);
  const [phoneLinked, setPhoneLinked] = useState(false);
  const viewport = useWindowDimensions();
  const hudScale = clamp(Math.min(viewport.width / 880, viewport.height / 400), 0.8, 2.2);

  const gameRef = useRef<Game | null>(null);
  const frameRef = useRef(0);
  const disposeRef = useRef<(() => void) | null>(null);
  const scratchRef = useRef<Scratch | null>(null);
  if (!scratchRef.current) scratchRef.current = createScratch();
  const sensorsRef = useRef<SensorState>({
    hasGyro: false,
    hasAccel: false,
    rawGyroX: 0,
    rawGyroZ: 0,
    rawUpX: 0.7,
    rawUpY: 0,
    rawUpZ: 0.7,
    gyroRate: 0,
    accelRoll: 0,
    upX: 0.7,
    upY: 0,
    upZ: 0.7,
    tilt: 0,
    touchLeft: false,
    touchRight: false,
    remoteActive: false,
    remoteSteer: 0,
  });

  useEffect(() => {
    const sensors = sensorsRef.current;
    let cancelled = false;
    let gyroSubscription: { remove: () => void } | null = null;
    let accelSubscription: { remove: () => void } | null = null;

    (async () => {
      const [hasGyro, hasAccel] = await Promise.all([
        Gyroscope.isAvailableAsync().catch(() => false),
        Accelerometer.isAvailableAsync().catch(() => false),
      ]);
      if (cancelled) return;
      setGyroAvailable(hasGyro);

      if (hasGyro) {
        try {
          Gyroscope.setUpdateInterval(SENSOR_INTERVAL_MS);
          gyroSubscription = Gyroscope.addListener(({ x, z }) => {
            sensors.rawGyroX = x;
            sensors.rawGyroZ = z;
          });
          sensors.hasGyro = true;
        } catch {
          setGyroAvailable(false); // the touch buttons take over
        }
      }
      if (hasAccel) {
        // Gravity gives an absolute roll angle. It corrects gyro drift, and it is the
        // only tilt source on phones with no gyroscope, such as the Galaxy A14 4G.
        const upSign = Platform.OS === 'ios' ? -1 : 1; // Android reports +1 g pointing up
        try {
          Accelerometer.setUpdateInterval(SENSOR_INTERVAL_MS);
          accelSubscription = Accelerometer.addListener(({ x, y, z }) => {
            const length = Math.sqrt(x * x + y * y + z * z) || 1;
            sensors.rawUpX = (upSign * x) / length;
            sensors.rawUpY = (upSign * y) / length;
            sensors.rawUpZ = (upSign * z) / length;
          });
          sensors.hasAccel = true;
        } catch {
          sensors.hasAccel = false;
        }
      }
    })();

    return () => {
      cancelled = true;
      gyroSubscription?.remove();
      accelSubscription?.remove();
      sensors.hasGyro = false;
      sensors.hasAccel = false;
    };
  }, []);

  const sizeRef = useRef(size);
  sizeRef.current = size;

  const qualityRef = useRef(quality);
  qualityRef.current = quality;

  const onContextCreate = useCallback((gl: ExpoWebGLRenderingContext) => {
    let bufferWidth = gl.drawingBufferWidth;
    let bufferHeight = gl.drawingBufferHeight;
    const tier = QUALITY[qualityRef.current];
    const RENDER_PIXEL_RATIO = renderPixelRatio(qualityRef.current);

    // three only needs a canvas-shaped object; the context comes from expo-gl.
    const canvas = {
      width: bufferWidth,
      height: bufferHeight,
      clientWidth: bufferWidth,
      clientHeight: bufferHeight,
      style: {},
      addEventListener: () => {},
      removeEventListener: () => {},
      getContext: () => gl,
    } as unknown as HTMLCanvasElement;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      context: gl as unknown as WebGLRenderingContext,
      antialias: false,
      alpha: false,
      stencil: false,
      depth: true,
    });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.shadowMap.enabled = tier.shadowSize > 0;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.setPixelRatio(RENDER_PIXEL_RATIO);
    renderer.setSize(bufferWidth / RENDER_PIXEL_RATIO, bufferHeight / RENDER_PIXEL_RATIO, false);

    renderer.setClearColor(SKY_HORIZON, 1);
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(SKY_HORIZON, 105, 330);

    // Exactly one ambient and one directional light, with no shadow maps.
    scene.add(new THREE.AmbientLight(0xbcd2f2, 1.55));
    const sun = new THREE.DirectionalLight(0xfff0d6, 2.35);
    sun.position.set(0.55, 0.75, 0.35);
    sun.castShadow = tier.shadowSize > 0;
    if (tier.shadowSize > 0) {
      sun.shadow.mapSize.set(tier.shadowSize, tier.shadowSize);
      const box = 46;
      sun.shadow.camera.left = -box;
      sun.shadow.camera.right = box;
      sun.shadow.camera.top = box;
      sun.shadow.camera.bottom = -box;
      sun.shadow.camera.near = 1;
      sun.shadow.camera.far = 180;
      sun.shadow.bias = -0.0006;
      sun.shadow.normalBias = 0.03;
    }
    scene.add(sun);
    scene.add(sun.target);

    const camera = new THREE.PerspectiveCamera(60, bufferWidth / bufferHeight, 0.5, 360);

    const game = createGame(scene, camera, scratchRef.current!, sensorsRef.current, {
      setRank,
      setLap,
      setSpeed,
      setItem,
      setPhase,
      setCountdown,
      setFinishTime,
      setProgress,
      setClock,
      setStandings,
    }, Math.min(tier.shadowSize > 0 ? 8 : 4, renderer.capabilities.getMaxAnisotropy() || 1), tier, renderer, sun);
    gameRef.current = game;

    let previous = 0;
    let perfTimer = 0;
    let perfFrames = 0;
    const loop = (now: number) => {
      frameRef.current = requestAnimationFrame(loop);
      const dt = previous ? Math.min(0.05, (now - previous) / 1000) : 1 / 60;
      previous = now;

      // The browser window can change size; the phone's orientation is locked.
      if (gl.drawingBufferWidth !== bufferWidth || gl.drawingBufferHeight !== bufferHeight) {
        bufferWidth = gl.drawingBufferWidth;
        bufferHeight = gl.drawingBufferHeight;
        renderer.setSize(bufferWidth / RENDER_PIXEL_RATIO, bufferHeight / RENDER_PIXEL_RATIO, false);
        camera.aspect = bufferWidth / bufferHeight;
        camera.updateProjectionMatrix();
      }

      updateRace(game, dt);
      syncKarts(game);
      syncItems(game);
      updateParticles(game, dt);
      placeCamera(game, Math.min(1, 9 * dt));

      renderer.render(scene, camera);
      gl.endFrameEXP();

      perfFrames++;
      perfTimer += dt;
      if (perfTimer >= 1) {
        setFps(Math.round(perfFrames / perfTimer));
        if (__DEV__) {
          const { triangles, calls } = renderer.info.render;
          console.log(`[kart] ${Math.round(perfFrames / perfTimer)} fps, ${triangles} tris, ${calls} draw calls`);
        }
        perfTimer = 0;
        perfFrames = 0;
      }
    };
    disposeRef.current = () => {
      scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
      });
      renderer.dispose();
    };
    frameRef.current = requestAnimationFrame(loop);
  }, []);

  const leaveRace = useCallback(() => {
    cancelAnimationFrame(frameRef.current);
    disposeRef.current?.();
    disposeRef.current = null;
    gameRef.current = null;
    setPhase('countdown');
    setScreen('menu');
  }, []);

  useEffect(() => () => {
    cancelAnimationFrame(frameRef.current);
    disposeRef.current?.();
  }, []);

  // On the PC the game also listens for the keyboard and for a phone acting as the wheel.
  useEffect(() => {
    if (!IS_WEB || screen !== 'race') return;
    const sensors = sensorsRef.current;
    const onKey = (down: boolean) => (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft') sensors.touchLeft = down;
      else if (event.key === 'ArrowRight') sensors.touchRight = down;
      else if (down && event.key === ' ' && gameRef.current) fireHeldItem(gameRef.current);
      else if (down && event.key === 'Enter' && gameRef.current?.phase === 'finished') resetRace(gameRef.current);
    };
    const keyDown = onKey(true);
    const keyUp = onKey(false);
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);

    let socket: WebSocket | null = null;
    let closed = false;
    let reconnect: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      socket = new WebSocket(`ws://${window.location.host}/ws?role=pc`);
      socket.onmessage = (event) => {
        const text = String(event.data);
        const kind = text[0];
        const g = gameRef.current;
        if (kind === 's') {
          sensors.remoteSteer = Number(text.slice(1)) || 0;
          sensors.remoteActive = true;
        } else if (kind === 'i' && g) fireHeldItem(g);
        else if (kind === 'a' && g?.phase === 'finished') resetRace(g);
        else if (kind === 'p') socket?.send(`q${text.slice(1)}`); // latency probe
        else if (kind === 'L') {
          const linked = text[1] === '1';
          setPhoneLinked(linked);
          if (!linked) {
            sensors.remoteActive = false;
            sensors.remoteSteer = 0;
          }
        }
      };
      socket.onclose = () => {
        setPhoneLinked(false);
        sensors.remoteActive = false;
        if (!closed) reconnect = setTimeout(connect, 1500);
      };
    };
    // A page opened without the PC server simply keeps its keyboard controls.
    if (window.location.protocol === 'http:') connect();
    const status = setInterval(() => {
      const g = gameRef.current;
      if (!g || socket?.readyState !== 1) return;
      socket.send(`S${JSON.stringify({ rank: g.shownRank, lap: g.shownLap, speed: g.shownSpeed, item: g.item, phase: g.phase })}`);
    }, 200);
    return () => {
      closed = true;
      clearInterval(status);
      if (reconnect) clearTimeout(reconnect);
      socket?.close();
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
    };
  }, [screen]);

  useEffect(() => {
    if (Platform.OS === 'android') NavigationBar.setHidden(true); // full-screen game
  }, []);

  const onLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    // The GL surface is created once, so wait for the locked landscape layout (see app.json).
    if (IS_WEB) setSize({ width, height });
    else if (width > height) setSize((current) => current ?? { width, height });
  }, []);

  const onItemPress = () => {
    if (gameRef.current) fireHeldItem(gameRef.current);
  };
  const playAgain = () => {
    if (gameRef.current) resetRace(gameRef.current);
  };
  const setTouch = (side: 'touchLeft' | 'touchRight', pressed: boolean) => {
    sensorsRef.current[side] = pressed;
  };

  if (screen === 'menu') {
    return (
      <MenuScreen
        onLayout={onLayout}
        quality={quality}
        onQuality={setQuality}
        onStart={() => setScreen('race')}
        onController={() => setScreen('controller')}
      />
    );
  }
  if (screen === 'controller') {
    return <ControllerScreen sensors={sensorsRef.current} onBack={() => setScreen('menu')} />;
  }

  const won = rank === 1;
  const showSteerButtons = !gyroAvailable && !IS_WEB;

  return (
    <View style={styles.root} onLayout={onLayout}>
      <StatusBar hidden />
      {size && (
        <GameSurface
          width={size.width}
          height={size.height}
          bufferScale={renderPixelRatio(quality) / DEVICE_PIXEL_RATIO}
          onContextCreate={onContextCreate}
        />
      )}

      <View
        pointerEvents={phase === 'finished' ? 'none' : 'box-none'}
        style={[
          styles.hud,
          phase === 'finished' && styles.hudHidden,
          {
            width: viewport.width / hudScale,
            height: viewport.height / hudScale,
            transform: [{ scale: hudScale }],
            transformOrigin: '0 0',
          },
        ]}
      >
        <RaceHud
          rank={rank}
          lap={lap}
          progress={progress}
          clockTenths={clockTenths}
          speed={speed}
          status={`${fps} FPS · ${QUALITY[quality].label}${IS_WEB ? (phoneLinked ? ' · Phone connected' : ' · Waiting for phone') : ''}`}
        />

        <View style={styles.bottomRow} pointerEvents="box-none">
          {showSteerButtons ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Steer left"
              onPressIn={() => setTouch('touchLeft', true)}
              onPressOut={() => setTouch('touchLeft', false)}
              style={({ pressed }) => [styles.steerButton, pressed && styles.steerButtonPressed]}
            >
              <Text allowFontScaling={false} style={styles.steerGlyph}>‹</Text>
            </Pressable>
          ) : (
            <View style={styles.steerSpacer} />
          )}

          <View style={styles.rightControls} pointerEvents="box-none">
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={item ? `Use ${item}` : 'Item slot, empty'}
              disabled={!item}
              onPress={onItemPress}
              style={[styles.itemSlot, item && styles.itemSlotFilled]}
            >
              {item ? (
                <Text allowFontScaling={false} style={styles.itemIcon}>{ITEM_ICON[item]}</Text>
              ) : (
                <Text allowFontScaling={false} style={styles.itemEmpty}>ITEM</Text>
              )}
            </Pressable>

            {showSteerButtons ? (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Steer right"
                onPressIn={() => setTouch('touchRight', true)}
                onPressOut={() => setTouch('touchRight', false)}
                style={({ pressed }) => [styles.steerButton, pressed && styles.steerButtonPressed]}
              >
                <Text allowFontScaling={false} style={styles.steerGlyph}>›</Text>
              </Pressable>
            ) : (
              IS_WEB ? <Speedometer speed={speed} /> : <View style={styles.steerSpacer} />
            )}
          </View>
        </View>
      </View>

      {(phase === 'countdown' || (phase === 'racing' && clockTenths < 9)) && (
        <View style={styles.centerOverlay} pointerEvents="none">
          <View
            style={[
              styles.countPlate,
              { transform: [{ translateY: -62 * hudScale }, { scale: hudScale }] },
              phase === 'racing' && styles.countPlateGo,
            ]}
          >
            <SlantPlate color={MENU_ACCENT} edge="#ffffff" />
            <Text allowFontScaling={false} style={[styles.countdown, phase === 'racing' && styles.countdownGo]}>
              {phase === 'racing' ? 'GO!' : Math.max(1, countdown)}
            </Text>
          </View>
          {phase === 'countdown' && (
            <Text allowFontScaling={false} style={[styles.hint, { marginTop: 8 * hudScale, fontSize: 12 * hudScale }]}>
              {IS_WEB ? 'TILT YOUR PHONE OR USE ← →  ·  SPACE TO USE ITEM' : gyroAvailable ? 'TILT YOUR PHONE TO STEER' : 'TILT YOUR PHONE, OR HOLD ‹ › TO STEER'}
            </Text>
          )}
        </View>
      )}

      {phase === 'finished' && (
        <ResultsPanel
          won={won}
          rank={rank}
          time={finishTime}
          standings={standings}
          scale={hudScale}
          width={viewport.width}
          height={viewport.height}
          onAgain={playAgain}
          onMenu={leaveRace}
        />
      )}
    </View>
  );
}

const MENU_ACCENT = '#ff1f8f';
const GAUGE_SIZE = 112;
const GAUGE_MAX_KMH = 200;
const GAUGE_TICKS = 25;
const GAUGE_REDLINE = 0.72;

/** Each tick re-renders only when it lights up or goes out, so speed changes stay cheap. */
const GaugeTick = memo(function GaugeTick({ index, lit }: { index: number; lit: boolean }) {
  const fraction = index / (GAUGE_TICKS - 1);
  const major = index % 4 === 0;
  const hot = fraction >= GAUGE_REDLINE;
  const color = lit ? (hot ? MENU_ACCENT : '#ffffff') : 'rgba(255,255,255,0.2)';
  return (
    <View style={[styles.tickArm, { transform: [{ rotate: `${-130 + fraction * 260}deg` }] }]}>
      <View style={[styles.tick, major && styles.tickMajor, { backgroundColor: color }]} />
    </View>
  );
});

const TICK_INDEXES = Array.from({ length: GAUGE_TICKS }, (_, i) => i);

const Speedometer = memo(function Speedometer({ speed }: { speed: number }) {
  const fraction = Math.min(1, speed / GAUGE_MAX_KMH);
  return (
    <View style={styles.gauge}>
      {TICK_INDEXES.map((i) => (
        <GaugeTick key={i} index={i} lit={i / (GAUGE_TICKS - 1) <= fraction && speed > 0} />
      ))}
      <View style={[styles.needleArm, { transform: [{ rotate: `${-130 + fraction * 260}deg` }] }]}>
        <View style={[styles.needle, fraction >= GAUGE_REDLINE && styles.needleHot]} />
      </View>
      <View style={styles.gaugeHub} />
      <View style={styles.gaugeReadout}>
        <Text allowFontScaling={false} style={styles.gaugeValue}>{speed}</Text>
        <Text allowFontScaling={false} style={styles.gaugeUnit}>km/h</Text>
      </View>
    </View>
  );
});

function formatRaceClock(tenths: number) {
  const totalSeconds = Math.floor(tenths / 10);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}.${tenths % 10}`;
}

/** Race standing and lap panels. Memoised so a speed change never redraws them. */
const StandingPanel = memo(function StandingPanel({ rank, lap, tenths }: { rank: number; lap: number; tenths: number }) {
  const label = PLACE_LABEL[rank - 1];
  return (
    <View style={styles.standing}>
      <View style={styles.posPanel}>
        <View style={styles.posPlate} />
        <Text allowFontScaling={false} style={styles.posNumber}>{label.slice(0, 1)}</Text>
        <Text allowFontScaling={false} style={styles.posSuffix}>{label.slice(1).toUpperCase()}</Text>
        <Text allowFontScaling={false} style={styles.posTotal}>/ 4</Text>
      </View>
      <View style={styles.raceRow}>
        <View style={styles.racePlateWrap}>
          <View style={styles.racePlate} />
          <Text allowFontScaling={false} style={styles.lapLabel}>{`Lap ${lap} / ${LAPS}`}</Text>
          <View style={styles.lapPips}>
            {[1, 2, 3].map((n) => (
              <View key={n} style={[styles.lapPip, n < lap && styles.lapPipDone, n === lap && styles.lapPipNow]} />
            ))}
          </View>
        </View>
        <View style={styles.racePlateWrap}>
          <View style={styles.racePlate} />
          <Text allowFontScaling={false} style={styles.raceClock}>{formatRaceClock(tenths)}</Text>
        </View>
      </View>
    </View>
  );
});

const ProgressTrack = memo(function ProgressTrack({ progress }: { progress: number }) {
  return (
    <View style={styles.progressWrap}>
      <View style={styles.progressHeader}>
        <Text allowFontScaling={false} style={styles.progressLabel}>PROGRESS</Text>
        <Text allowFontScaling={false} style={styles.progressValue}>{`${progress}%`}</Text>
      </View>
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${progress}%` }]} />
        <View style={[styles.progressLapTick, { left: '33.33%' }]} />
        <View style={[styles.progressLapTick, { left: '66.66%' }]} />
      </View>
    </View>
  );
});

function RaceHud({
  rank,
  lap,
  progress,
  clockTenths,
  speed,
  status,
}: {
  rank: number;
  lap: number;
  progress: number;
  clockTenths: number;
  speed: number;
  status: string;
}) {
  return (
    <View style={styles.topRow} pointerEvents="none">
      <StandingPanel rank={rank} lap={lap} tenths={clockTenths} />
      <ProgressTrack progress={progress} />
      <View style={styles.speedCorner}>
        {!IS_WEB && <Speedometer speed={speed} />}
        <Text allowFontScaling={false} style={styles.fpsText}>{status}</Text>
      </View>
    </View>
  );
}

// Relative trade-off between the tiers, out of eight segments.
const TIER_STATS: Record<Quality, { visuals: number; smooth: number }> = {
  low: { visuals: 2, smooth: 8 },
  medium: { visuals: 4, smooth: 7 },
  high: { visuals: 6, smooth: 5 },
  ultra: { visuals: 8, smooth: 3 },
};

function StatBar({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.statRow}>
      <Text allowFontScaling={false} style={styles.statLabel}>{label}</Text>
      <View style={styles.statSegments}>
        {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
          <View key={n} style={[styles.statSegment, n <= value && styles.statSegmentOn]} />
        ))}
      </View>
    </View>
  );
}

/** A slanted plate behind unslanted content: the shape every menu tile is built from. */
function SlantPlate({ color, edge }: { color: string; edge?: string }) {
  return <View style={[styles.slantPlate, { backgroundColor: color }, edge ? { borderBottomWidth: 3, borderBottomColor: edge } : null]} />;
}

function QualityTile({ value, selected, onPress }: { value: Quality; selected: boolean; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected }}
      accessibilityLabel={`${QUALITY[value].label} graphics`}
      onPress={onPress}
      style={({ pressed }) => [styles.qualityTile, pressed && styles.tilePressed]}
    >
      <SlantPlate color={selected ? MENU_ACCENT : 'rgba(10,14,24,0.74)'} edge={selected ? '#ffffff' : 'rgba(255,255,255,0.24)'} />
      <Text allowFontScaling={false} style={[styles.qualityText, selected && styles.qualityTextOn]}>
        {QUALITY[value].label.toUpperCase()}
      </Text>
    </Pressable>
  );
}

function MenuScreen({
  onLayout,
  quality,
  onQuality,
  onStart,
  onController,
}: {
  onLayout: (event: LayoutChangeEvent) => void;
  quality: Quality;
  onQuality: (q: Quality) => void;
  onStart: () => void;
  onController: () => void;
}) {
  const order: Quality[] = ['low', 'medium', 'high', 'ultra'];
  const [links, setLinks] = useState<{ usb: string; wifi: string[] } | null>(null);
  useEffect(() => {
    if (!IS_WEB) return;
    fetch('/info')
      .then((response) => (response.ok ? response.json() : null))
      .then(setLinks)
      .catch(() => {});
  }, []);
  // The layout is drawn for a phone in landscape and scaled up as one piece on bigger screens.
  const window = useWindowDimensions();
  const scale = clamp(Math.min(window.width / 880, window.height / 400), 0.8, 2.2);
  const stats = TIER_STATS[quality];
  return (
    <View style={styles.menu} onLayout={onLayout}>
      <StatusBar hidden />
      <Image source={require('./assets/menu-hero.jpg')} style={styles.menuHero} resizeMode="cover" />
      <View
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: window.width / scale,
          height: window.height / scale,
          transform: [{ scale }],
          transformOrigin: '0 0',
          paddingHorizontal: 30,
          paddingVertical: 20,
          justifyContent: 'space-between',
        }}
      >
        <View>
          <Text allowFontScaling={false} style={styles.logo}>CARJY</Text>
          <View style={styles.logoRule}>
            <View style={styles.logoSlash} />
            <Text allowFontScaling={false} style={styles.logoSub}>RACING</Text>
          </View>
        </View>

        <View style={styles.menuBody}>
          <Text allowFontScaling={false} style={styles.menuSection}>GRAPHICS QUALITY</Text>
          <View style={styles.qualityRow}>
            {order.map((q) => (
              <QualityTile key={q} value={q} selected={q === quality} onPress={() => onQuality(q)} />
            ))}
          </View>
          <View style={styles.statBlock}>
            <StatBar label="VISUALS" value={stats.visuals} />
            <StatBar label="SMOOTHNESS" value={stats.smooth} />
            <Text allowFontScaling={false} style={styles.qualityNote}>{QUALITY[quality].note}</Text>
          </View>
        </View>

        <View>
          <View style={styles.menuActions}>
            <Pressable accessibilityRole="button" onPress={onStart} style={({ pressed }) => [styles.actionTile, styles.actionPrimary, pressed && styles.tilePressed]}>
              <SlantPlate color={MENU_ACCENT} edge="#ffffff" />
              <View>
                <Text allowFontScaling={false} style={styles.actionTitle}>{IS_WEB ? 'START RACE' : 'PLAY ON PHONE'}</Text>
                <Text allowFontScaling={false} style={styles.actionSub}>{`${LAPS} LAPS  ·  4 RACERS`}</Text>
              </View>
              <Text allowFontScaling={false} style={styles.actionArrow}>›</Text>
            </Pressable>
            {!IS_WEB && (
              <Pressable accessibilityRole="button" onPress={onController} style={({ pressed }) => [styles.actionTile, pressed && styles.tilePressed]}>
                <SlantPlate color="rgba(10,14,24,0.8)" edge="rgba(255,255,255,0.32)" />
                <View>
                  <Text allowFontScaling={false} style={styles.actionTitle}>PLAY ON PC</Text>
                  <Text allowFontScaling={false} style={styles.actionSub}>PHONE AS WHEEL</Text>
                </View>
                <Text allowFontScaling={false} style={styles.actionArrow}>›</Text>
              </Pressable>
            )}
          </View>
          {links && (
            <View style={styles.linkBox}>
              <Text allowFontScaling={false} style={styles.linkLine}>
                <Text style={styles.linkTitle}>PHONE WHEEL  </Text>
                {`USB ${links.usb}${links.wifi.map((address) => `   Wi-Fi ${address}`).join('')}`}
              </Text>
              <Text allowFontScaling={false} style={styles.linkHint}>Open Carjy on your phone, choose Play on PC, then enter an address  ·  Keyboard: ← → steer, Space item, F11 fullscreen</Text>
            </View>
          )}
        </View>
      </View>
    </View>
  );
}

function ResultsPanel({
  won,
  rank,
  time,
  standings,
  scale,
  width,
  height,
  onAgain,
  onMenu,
}: {
  won: boolean;
  rank: number;
  time: number;
  standings: StandingRow[];
  scale: number;
  width: number;
  height: number;
  onAgain: () => void;
  onMenu: () => void;
}) {
  return (
    <View style={styles.resultsScrim} accessibilityViewIsModal>
      <View
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: width / scale,
          height: height / scale,
          transform: [{ scale }],
          transformOrigin: '0 0',
          paddingHorizontal: 34,
          paddingVertical: 20,
          justifyContent: 'center',
        }}
      >
        <View style={styles.resultsTitle}>
          <SlantPlate color={won ? MENU_ACCENT : '#1a1e2c'} edge={won ? '#ffffff' : MENU_ACCENT} />
          <Text allowFontScaling={false} style={styles.resultsTitleText}>{won ? 'VICTORY!' : 'GAME OVER'}</Text>
        </View>
        <View style={styles.resultsRow}>
          <View style={styles.resultsSummary}>
            <Text allowFontScaling={false} style={styles.resultsLabel}>YOU FINISHED</Text>
            <Text allowFontScaling={false} style={styles.resultsPlace}>{`${PLACE_LABEL[rank - 1].toUpperCase()} OF 4`}</Text>
            <Text allowFontScaling={false} style={[styles.resultsLabel, styles.resultsLabelGap]}>RACE TIME</Text>
            <Text allowFontScaling={false} style={styles.resultsTime}>{formatRaceClock(Math.round(time * 10))}</Text>
          </View>
          <View style={styles.resultsTable}>
            {standings.map((row, index) => (
              <View key={row.name} style={[styles.standingRow, row.isPlayer && styles.standingRowYou]}>
                <Text allowFontScaling={false} style={styles.standingPlace}>{index + 1}</Text>
                <View style={[styles.standingSwatch, { backgroundColor: row.color }]} />
                <Text allowFontScaling={false} style={styles.standingName}>{row.name}</Text>
              </View>
            ))}
          </View>
        </View>
        <View style={styles.resultsActions}>
          <Pressable accessibilityRole="button" onPress={onAgain} style={({ pressed }) => [styles.actionTile, styles.actionPrimary, pressed && styles.tilePressed]}>
            <SlantPlate color={MENU_ACCENT} edge="#ffffff" />
            <Text allowFontScaling={false} style={styles.actionTitle}>PLAY AGAIN</Text>
            <Text allowFontScaling={false} style={styles.actionArrow}>›</Text>
          </Pressable>
          <Pressable accessibilityRole="button" onPress={onMenu} style={({ pressed }) => [styles.actionTile, pressed && styles.tilePressed]}>
            <SlantPlate color="rgba(10,14,24,0.86)" edge="rgba(255,255,255,0.32)" />
            <Text allowFontScaling={false} style={styles.actionTitle}>MAIN MENU</Text>
            <Text allowFontScaling={false} style={styles.actionArrow}>›</Text>
          </Pressable>
        </View>
        <Text allowFontScaling={false} style={styles.credit}>
          Car Concept by Eric Chadwick, © 2024 Darmstadt Graphics Group, CC BY 4.0, modified
        </Text>
      </View>
    </View>
  );
}

/** Phone as a steering wheel: reads the same tilt pipeline and sends it to the PC. */
function ControllerScreen({ sensors, onBack }: { sensors: SensorState; onBack: () => void }) {
  const [host, setHost] = useState(`127.0.0.1:${REMOTE_PORT}`);
  const [status, setStatus] = useState<'off' | 'connecting' | 'on' | 'failed'>('off');
  const [latency, setLatency] = useState<number | null>(null);
  const [steerShown, setSteerShown] = useState(0);
  const [race, setRace] = useState<{ rank: number; lap: number; speed: number; item: ItemKind | null; phase: Phase } | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const viewport = useWindowDimensions();
  const scale = clamp(Math.min(viewport.width / 880, viewport.height / 400), 0.8, 2.2);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
  }, []);

  const connect = useCallback(() => {
    disconnect();
    setStatus('connecting');
    const socket = new WebSocket(`ws://${host.trim()}/ws?role=phone`);
    socketRef.current = socket;
    socket.onopen = () => setStatus('on');
    socket.onerror = () => setStatus('failed');
    socket.onclose = () => {
      setRace(null);
      setLatency(null);
      setStatus((current) => (current === 'on' ? 'off' : current === 'connecting' ? 'failed' : current));
    };
    socket.onmessage = (event) => {
      const text = String(event.data);
      if (text[0] === 'S') setRace(JSON.parse(text.slice(1)));
      else if (text[0] === 'q') setLatency(Date.now() - Number(text.slice(1)));
    };
  }, [disconnect, host]);

  useEffect(() => {
    let frame = 0;
    let previous = 0;
    let shown = 0;
    const loop = (now: number) => {
      frame = requestAnimationFrame(loop);
      const dt = previous ? Math.min(0.05, (now - previous) / 1000) : 1 / 60;
      previous = now;
      const steer = readSensorSteering(sensors, dt);
      const socket = socketRef.current;
      if (socket?.readyState === 1) socket.send(`s${steer.toFixed(3)}`);
      if (++shown % 6 === 0) setSteerShown(steer);
    };
    frame = requestAnimationFrame(loop);
    const ping = setInterval(() => {
      if (socketRef.current?.readyState === 1) socketRef.current.send(`p${Date.now()}`);
    }, 1000);
    return () => {
      cancelAnimationFrame(frame);
      clearInterval(ping);
      socketRef.current?.close();
      sensors.touchLeft = false;
      sensors.touchRight = false;
    };
  }, [sensors]);

  const statusText = { off: 'NOT CONNECTED', connecting: 'CONNECTING...', on: 'CONNECTED', failed: 'COULD NOT CONNECT. CHECK THE ADDRESS AND THAT CARJY IS OPEN ON THE PC.' }[status];
  const finished = race?.phase === 'finished';
  const wheelPosition = Math.round(((steerShown + 1) / 2) * 20);
  return (
    <View style={styles.menu}>
      <StatusBar hidden />
      <Image source={require('./assets/menu-hero.jpg')} style={[styles.menuHero, styles.controllerHero]} resizeMode="cover" />
      <View
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: viewport.width / scale,
          height: viewport.height / scale,
          transform: [{ scale }],
          transformOrigin: '0 0',
          paddingHorizontal: 22,
          paddingVertical: 14,
          justifyContent: 'space-between',
        }}
      >
        <View style={styles.controllerTop}>
          <Pressable accessibilityRole="button" onPress={onBack} style={({ pressed }) => [styles.controllerBack, pressed && styles.tilePressed]}>
            <SlantPlate color="rgba(10,14,24,0.86)" edge="rgba(255,255,255,0.32)" />
            <Text allowFontScaling={false} style={styles.controllerBackText}>‹ MENU</Text>
          </Pressable>
          <View style={styles.hostWrap}>
            <SlantPlate color="rgba(10,14,24,0.86)" edge="rgba(255,255,255,0.32)" />
            <Text allowFontScaling={false} style={styles.hostLabel}>PC ADDRESS</Text>
            <TextInput
              value={host}
              onChangeText={setHost}
              autoCapitalize="none"
              autoCorrect={false}
              accessibilityLabel="PC address"
              style={styles.hostInput}
              allowFontScaling={false}
            />
          </View>
          <Pressable accessibilityRole="button" accessibilityLabel="Connect" onPress={connect} style={({ pressed }) => [styles.connectButton, pressed && styles.tilePressed]}>
            <SlantPlate color={MENU_ACCENT} edge="#ffffff" />
            <Text allowFontScaling={false} style={styles.connectText}>CONNECT</Text>
          </Pressable>
        </View>

        <View style={styles.controllerMiddle}>
          <View style={styles.controllerInfo}>
            <View style={[styles.statusDot, status === 'on' && styles.statusDotOn, status === 'failed' && styles.statusDotFail]} />
            <Text allowFontScaling={false} style={styles.controllerStatus}>
              {statusText}{status === 'on' && latency !== null ? `  ·  ${latency} MS LATENCY` : ''}
              {race ? `  ·  ${PLACE_LABEL[race.rank - 1].toUpperCase()} / 4  ·  LAP ${race.lap} / ${LAPS}  ·  ${race.speed} KM/H` : ''}
            </Text>
          </View>
          <View style={styles.wheelMeter}>
            {Array.from({ length: 21 }, (_, i) => {
              const centre = i === 10;
              const between = (i >= 10 && i <= wheelPosition) || (i <= 10 && i >= wheelPosition);
              return <View key={i} style={[styles.wheelTick, centre && styles.wheelTickCentre, between && styles.wheelTickOn]} />;
            })}
          </View>
          <Text allowFontScaling={false} style={styles.wheelLabel}>TILT TO STEER</Text>
        </View>

        <View style={styles.controllerButtons}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Steer left"
            onPressIn={() => { sensors.touchLeft = true; }}
            onPressOut={() => { sensors.touchLeft = false; }}
            style={({ pressed }) => [styles.steerButton, pressed && styles.steerButtonPressed]}
          >
            <Text allowFontScaling={false} style={styles.steerGlyph}>‹</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={finished ? 'Play again' : 'Use item'}
            onPress={() => socketRef.current?.send(finished ? 'a' : 'i')}
            style={({ pressed }) => [styles.controllerItem, pressed && styles.tilePressed]}
          >
            <SlantPlate color={finished || race?.item ? MENU_ACCENT : 'rgba(10,14,24,0.86)'} edge={finished || race?.item ? '#ffffff' : 'rgba(255,255,255,0.32)'} />
            <Text allowFontScaling={false} style={race?.item && !finished ? styles.controllerItemIcon : styles.controllerItemText}>
              {finished ? 'PLAY AGAIN' : race?.item ? ITEM_ICON[race.item] : 'ITEM'}
            </Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Steer right"
            onPressIn={() => { sensors.touchRight = true; }}
            onPressOut={() => { sensors.touchRight = false; }}
            style={({ pressed }) => [styles.steerButton, pressed && styles.steerButtonPressed]}
          >
            <Text allowFontScaling={false} style={styles.steerGlyph}>›</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const FILL = { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0 } as const;

const styles = StyleSheet.create({
  hudHidden: { opacity: 0 },
  centerOverlay: { ...FILL, alignItems: 'center', justifyContent: 'center' },
  countPlate: { width: 92, height: 80, alignItems: 'center', justifyContent: 'center', position: 'absolute' },
  countPlateGo: { width: 136 },
  countdown: { color: '#ffffff', fontSize: 58, lineHeight: 68, fontWeight: '900', fontStyle: 'italic' },
  countdownGo: { fontSize: 44, lineHeight: 54 },
  hint: { color: '#ffffff', fontWeight: '800', fontStyle: 'italic', letterSpacing: 1, backgroundColor: 'rgba(8,11,20,0.76)', paddingHorizontal: 12, paddingVertical: 4 },
  resultsScrim: { ...FILL, backgroundColor: 'rgba(6,8,16,0.68)' },
  resultsTitle: { alignSelf: 'flex-start', height: 62, justifyContent: 'center', paddingHorizontal: 24 },
  resultsTitleText: { color: '#ffffff', fontSize: 40, lineHeight: 48, fontWeight: '900', fontStyle: 'italic', letterSpacing: 0.5 },
  resultsRow: { flexDirection: 'row', gap: 28, marginTop: 14, alignItems: 'flex-start' },
  resultsSummary: { width: 190 },
  resultsLabel: { color: 'rgba(255,255,255,0.72)', fontSize: 10, fontWeight: '800', letterSpacing: 1.5 },
  resultsLabelGap: { marginTop: 8 },
  resultsPlace: { color: '#ffffff', fontSize: 28, lineHeight: 33, fontWeight: '900', fontStyle: 'italic' },
  resultsTime: { color: '#ffffff', fontSize: 28, lineHeight: 33, fontWeight: '900', fontStyle: 'italic', fontVariant: ['tabular-nums'] },
  resultsTable: { width: 260, gap: 3 },
  standingRow: { flexDirection: 'row', alignItems: 'center', gap: 10, height: 24, paddingHorizontal: 10, backgroundColor: 'rgba(10,14,24,0.74)', transform: [{ skewX: '-10deg' }] },
  standingRowYou: { backgroundColor: MENU_ACCENT },
  standingPlace: { width: 14, color: '#ffffff', fontSize: 14, fontWeight: '900', fontStyle: 'italic', fontVariant: ['tabular-nums'] },
  standingSwatch: { width: 18, height: 9 },
  standingName: { color: '#ffffff', fontSize: 13, fontWeight: '800', fontStyle: 'italic', letterSpacing: 0.5 },
  resultsActions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  credit: { marginTop: 8, fontSize: 9, color: 'rgba(255,255,255,0.68)' },
  controllerHero: { opacity: 0.3 },
  controllerTop: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  controllerBack: { width: 92, height: 44, alignItems: 'center', justifyContent: 'center' },
  controllerBackText: { color: '#ffffff', fontSize: 14, fontWeight: '800', fontStyle: 'italic' },
  hostWrap: { flex: 1, height: 44, flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 18 },
  hostLabel: { color: 'rgba(255,255,255,0.68)', fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  hostInput: { flex: 1, height: 44, color: '#ffffff', fontSize: 15, fontWeight: '700', padding: 0 },
  connectButton: { width: 126, height: 44, alignItems: 'center', justifyContent: 'center' },
  connectText: { color: '#ffffff', fontSize: 15, fontWeight: '900', fontStyle: 'italic' },
  controllerMiddle: { alignItems: 'center' },
  controllerInfo: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 10, paddingHorizontal: 12 },
  statusDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.4)' },
  statusDotOn: { backgroundColor: '#35e07a' },
  statusDotFail: { backgroundColor: MENU_ACCENT },
  controllerStatus: { flexShrink: 1, color: '#ffffff', fontSize: 12, fontWeight: '700', fontStyle: 'italic', fontVariant: ['tabular-nums'], textAlign: 'center' },
  wheelMeter: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  wheelTick: { width: 12, height: 22, backgroundColor: 'rgba(255,255,255,0.18)', transform: [{ skewX: '-14deg' }] },
  wheelTickCentre: { height: 30, backgroundColor: 'rgba(255,255,255,0.72)' },
  wheelTickOn: { backgroundColor: MENU_ACCENT },
  wheelLabel: { color: 'rgba(255,255,255,0.72)', fontSize: 10, fontWeight: '800', letterSpacing: 2, marginTop: 7 },
  controllerButtons: { flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between' },
  controllerItem: { width: 180, height: 72, alignItems: 'center', justifyContent: 'center' },
  controllerItemText: { color: '#ffffff', fontSize: 17, fontWeight: '900', fontStyle: 'italic', letterSpacing: 0.5 },
  controllerItemIcon: { fontSize: 36, lineHeight: 46 },
  fpsText: { color: '#ffffff', fontSize: 10, fontWeight: '700', marginTop: 3, textShadowColor: 'rgba(0,0,0,0.8)', textShadowRadius: 3, textShadowOffset: { width: 0, height: 1 } },
  menu: { flex: 1, backgroundColor: '#0a0e18', overflow: 'hidden' },
  menuHero: { position: 'absolute', left: 0, top: 0, right: 0, bottom: 0, width: '100%', height: '100%' },
  logo: { color: '#ffffff', fontSize: 58, lineHeight: 60, fontWeight: '900', fontStyle: 'italic', letterSpacing: 1, textShadowColor: 'rgba(0,0,0,0.5)', textShadowRadius: 8, textShadowOffset: { width: 0, height: 2 } },
  logoRule: { flexDirection: 'row', alignItems: 'center', gap: 9, marginTop: -1 },
  logoSlash: { width: 62, height: 8, backgroundColor: '#ff1f8f', transform: [{ skewX: '-28deg' }] },
  logoSub: { color: '#ffffff', fontSize: 15, fontWeight: '800', fontStyle: 'italic', letterSpacing: 6 },
  menuBody: { alignItems: 'flex-start' },
  menuSection: { color: '#ffffff', fontSize: 11, fontWeight: '800', letterSpacing: 2, marginBottom: 6, opacity: 0.86 },
  qualityRow: { flexDirection: 'row', gap: 7 },
  qualityTile: { width: 84, height: 40, alignItems: 'center', justifyContent: 'center' },
  qualityText: { color: '#dfe7f0', fontSize: 13, fontWeight: '800', fontStyle: 'italic', letterSpacing: 0.5 },
  qualityTextOn: { color: '#ffffff', fontWeight: '900' },
  tilePressed: { opacity: 0.78, transform: [{ scale: 0.97 }] },
  slantPlate: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, transform: [{ skewX: '-14deg' }] },
  statBlock: { marginTop: 9, gap: 3 },
  statRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  statLabel: { width: 76, color: '#d6e0ea', fontSize: 9, fontWeight: '800', letterSpacing: 1 },
  statSegments: { flexDirection: 'row', gap: 3 },
  statSegment: { width: 15, height: 6, backgroundColor: 'rgba(255,255,255,0.2)', transform: [{ skewX: '-22deg' }] },
  statSegmentOn: { backgroundColor: '#ff1f8f' },
  qualityNote: { color: '#ffffff', fontSize: 11, fontWeight: '600', fontStyle: 'italic', marginTop: 3, opacity: 0.86 },
  menuActions: { flexDirection: 'row', gap: 12 },
  actionTile: { width: 204, height: 60, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingLeft: 20, paddingRight: 15 },
  actionPrimary: { width: 232 },
  actionTitle: { color: '#ffffff', fontSize: 18, lineHeight: 22, fontWeight: '900', fontStyle: 'italic', letterSpacing: 0.4 },
  actionSub: { color: 'rgba(255,255,255,0.86)', fontSize: 9, fontWeight: '800', letterSpacing: 1.2, marginTop: 1 },
  actionArrow: { color: '#ffffff', fontSize: 38, lineHeight: 42, fontWeight: '300' },
  linkBox: { marginTop: 9 },
  linkTitle: { color: '#ff5fb0', fontWeight: '900', letterSpacing: 1 },
  linkLine: { color: '#ffffff', fontSize: 11, fontWeight: '700', fontVariant: ['tabular-nums'] },
  linkHint: { color: 'rgba(255,255,255,0.72)', fontSize: 9, marginTop: 2 },
  root: { flex: 1, backgroundColor: '#bcdff5', overflow: 'hidden' },
  hud: { position: 'absolute', left: 0, top: 0, justifyContent: 'space-between', paddingVertical: 12, paddingHorizontal: 18 },
  topRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  standing: { width: 240, alignItems: 'flex-start' },
  posPanel: { flexDirection: 'row', alignItems: 'baseline', paddingLeft: 15, paddingRight: 20, height: 56 },
  posPlate: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, backgroundColor: MENU_ACCENT, borderBottomWidth: 3, borderBottomColor: '#ffffff', transform: [{ skewX: '-14deg' }] },
  posNumber: { color: '#ffffff', fontSize: 46, lineHeight: 54, fontWeight: '900', fontStyle: 'italic' },
  posSuffix: { color: '#ffffff', fontSize: 17, fontWeight: '900', fontStyle: 'italic', marginLeft: 2 },
  posTotal: { color: 'rgba(255,255,255,0.82)', fontSize: 16, fontWeight: '800', fontStyle: 'italic', marginLeft: 8 },
  raceRow: { flexDirection: 'row', gap: 6, marginTop: 6 },
  racePlateWrap: { flexDirection: 'row', alignItems: 'center', gap: 8, height: 26, paddingHorizontal: 11 },
  racePlate: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, backgroundColor: 'rgba(8,11,20,0.78)', borderBottomWidth: 2, borderBottomColor: 'rgba(255,255,255,0.3)', transform: [{ skewX: '-14deg' }] },
  lapLabel: { color: '#ffffff', fontSize: 13, fontWeight: '900', fontStyle: 'italic', fontVariant: ['tabular-nums'] },
  lapPips: { flexDirection: 'row', gap: 3 },
  lapPip: { width: 11, height: 5, backgroundColor: 'rgba(255,255,255,0.24)', transform: [{ skewX: '-22deg' }] },
  lapPipDone: { backgroundColor: '#ffffff' },
  lapPipNow: { backgroundColor: MENU_ACCENT },
  raceClock: { color: '#ffffff', fontSize: 13, fontWeight: '800', fontStyle: 'italic', fontVariant: ['tabular-nums'] },
  progressWrap: { width: 240, paddingTop: 3 },
  progressHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 3 },
  progressLabel: { color: '#ffffff', fontSize: 10, fontWeight: '800', fontStyle: 'italic', letterSpacing: 1, textShadowColor: 'rgba(0,0,0,0.85)', textShadowRadius: 3, textShadowOffset: { width: 0, height: 1 } },
  progressValue: { color: '#ffffff', fontSize: 12, fontWeight: '900', fontStyle: 'italic', fontVariant: ['tabular-nums'], textShadowColor: 'rgba(0,0,0,0.85)', textShadowRadius: 3, textShadowOffset: { width: 0, height: 1 } },
  progressTrack: { height: 7, backgroundColor: 'rgba(8,11,20,0.76)', overflow: 'hidden', transform: [{ skewX: '-20deg' }] },
  progressFill: { height: '100%', backgroundColor: MENU_ACCENT },
  progressLapTick: { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: 'rgba(255,255,255,0.75)' },
  speedCorner: { width: 240, alignItems: 'flex-end' },
  gauge: { width: GAUGE_SIZE, height: GAUGE_SIZE, borderRadius: GAUGE_SIZE / 2, backgroundColor: 'rgba(8,11,20,0.78)', borderWidth: 2, borderColor: 'rgba(255,255,255,0.3)' },
  tickArm: { position: 'absolute', left: GAUGE_SIZE / 2 - 3, top: 3, width: 2, height: GAUGE_SIZE - 10, alignItems: 'center' },
  tick: { width: 2, height: 7 },
  tickMajor: { width: 3, height: 11 },
  needleArm: { position: 'absolute', left: GAUGE_SIZE / 2 - 3.5, top: 15, width: 3, height: GAUGE_SIZE - 34, alignItems: 'center' },
  needle: { width: 3, height: (GAUGE_SIZE - 34) / 2, borderRadius: 2, backgroundColor: MENU_ACCENT },
  needleHot: { backgroundColor: '#ffffff' },
  gaugeHub: { position: 'absolute', left: GAUGE_SIZE / 2 - 8, top: GAUGE_SIZE / 2 - 8, width: 12, height: 12, borderRadius: 6, backgroundColor: '#e8eef5', borderWidth: 2, borderColor: '#101a26' },
  gaugeReadout: { position: 'absolute', left: 0, right: 0, bottom: 6, alignItems: 'center' },
  gaugeValue: { color: '#ffffff', fontSize: 24, lineHeight: 26, fontWeight: '900', fontStyle: 'italic', fontVariant: ['tabular-nums'] },
  gaugeUnit: { color: 'rgba(255,255,255,0.72)', fontSize: 9, lineHeight: 10, fontWeight: '800', letterSpacing: 0.5 },
  bottomRow: { flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between' },
  rightControls: { flexDirection: 'row', alignItems: 'flex-end', gap: 14 },
  steerButton: { width: 88, height: 84, backgroundColor: 'rgba(8,11,20,0.42)', borderBottomWidth: 3, borderBottomColor: 'rgba(255,255,255,0.55)', transform: [{ skewX: '-10deg' }], alignItems: 'center', justifyContent: 'center' },
  steerButtonPressed: { backgroundColor: MENU_ACCENT },
  steerSpacer: { width: 88 },
  steerGlyph: { color: '#ffffff', fontSize: 58, lineHeight: 66, fontWeight: '300' },
  itemSlot: { width: 76, height: 70, backgroundColor: 'rgba(8,11,20,0.58)', borderBottomWidth: 3, borderBottomColor: 'rgba(255,255,255,0.4)', transform: [{ skewX: '-10deg' }], alignItems: 'center', justifyContent: 'center' },
  itemSlotFilled: { borderBottomColor: '#ffffff', backgroundColor: MENU_ACCENT },
  itemIcon: { fontSize: 36, lineHeight: 46, transform: [{ skewX: '10deg' }] },
  itemEmpty: { color: 'rgba(255,255,255,0.72)', fontSize: 12, fontWeight: '800', fontStyle: 'italic', letterSpacing: 1 },
});
