// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const OceanArenaChallengers = (() => {
  const CHALLENGERS = Object.freeze([
    { name: "GLO12", color: "#4fd1ff" },
    { name: "GLONET", color: "#ff6b9d" },
    { name: "XiHe", color: "#ffd166" },
    { name: "WenHai", color: "#7ee081" },
  ]);

  const DIMENSIONS = Object.freeze({
    variable: [
      { key: "temperature", label: "🌡️ Temperature" },
      { key: "salinity", label: "🧂 Salinity" },
      { key: "height", label: "🌊 Sea level" },
      { key: "currents", label: "🌀 Currents" },
    ],
    horizon: [
      { key: "short", label: "1–3 days" },
      { key: "medium", label: "4–6 days" },
      { key: "long", label: "7–10 days" },
    ],
    depth: [
      { key: "surface", label: "Surface" },
      { key: "upper", label: "5–200 m" },
      { key: "intermediate", label: "200–600 m" },
    ],
  });

  const HEXAGON_RESOLUTION = 2;

  const _hashString = (text) =>
    [...text].reduce((hash, character) => Math.imul(hash ^ character.charCodeAt(0), 16777619), 2166136261) >>> 0;

  const _seededRandom = (seed) => (index) => {
    const value = Math.sin(seed * 12.9898 + index * 78.233) * 43758.5453;
    return value - Math.floor(value);
  };

  const _smoothField = (seed) => {
    const random = _seededRandom(seed);
    const waves = [0, 1, 2, 3].map((index) => ({
      latitudeFrequency: 1 + random(index * 4) * 3,
      longitudeFrequency: 1 + random(index * 4 + 1) * 3,
      phase: random(index * 4 + 2) * Math.PI * 2,
      amplitude: 0.5 + random(index * 4 + 3),
    }));
    return (latitude, longitude) =>
      waves.reduce(
        (sum, wave) =>
          sum +
          wave.amplitude *
            Math.sin((latitude * Math.PI / 180) * wave.latitudeFrequency + wave.phase) *
            Math.cos((longitude * Math.PI / 180) * wave.longitudeFrequency + wave.phase),
        0,
      );
  };

  /** Fake root mean square error of each challenger at a location, lower is better. */
  const fakeScores = (selection, latitude, longitude) =>
    CHALLENGERS.map((challenger) => ({
      challenger,
      error:
        1 +
        0.4 *
          _smoothField(_hashString(`${challenger.name}|${selection.variable}|${selection.horizon}|${selection.depth}`))(
            latitude,
            longitude,
          ),
    })).sort((first, second) => first.error - second.error);

  return Object.freeze({ CHALLENGERS, DIMENSIONS, HEXAGON_RESOLUTION, fakeScores });
})();
