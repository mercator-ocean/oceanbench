// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const EARTH_METRES_PER_DEGREE = 111_320;
const DAY_SECONDS = 86_400;
const SUBSTEPS_PER_DAY = 4;

function sample(field, longitude, latitude) {
  let lon = longitude;
  if (field.periodic) lon = ((((lon - field.lon0) % 360) + 360) % 360) + field.lon0;
  const fx = (lon - field.lon0) / field.lonStep;
  const fy = (latitude - field.lat0) / field.latStep;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  if (x0 < 0 || x0 >= field.width - 1 || y0 < 0 || y0 >= field.height - 1) return null;
  const offset = y0 * field.width + x0;
  const indices = [offset, offset + 1, offset + field.width, offset + field.width + 1];
  const values = indices.map((index) => [field.u[index], field.v[index]]);
  if (values.some(([u, v]) => !Number.isFinite(u) || !Number.isFinite(v))) return null;
  const tx = fx - x0;
  const ty = fy - y0;
  const interpolate = (component) => {
    const top = values[0][component] + (values[1][component] - values[0][component]) * tx;
    const bottom = values[2][component] + (values[3][component] - values[2][component]) * tx;
    return top + (bottom - top) * ty;
  };
  return { u: interpolate(0), v: interpolate(1) };
}

function displacement(position, velocity, seconds) {
  const latitudeRadians = position.latitude * Math.PI / 180;
  const longitudeScale = EARTH_METRES_PER_DEGREE * Math.max(0.05, Math.cos(latitudeRadians));
  return {
    longitude: position.longitude + velocity.u * seconds / longitudeScale,
    latitude: position.latitude + velocity.v * seconds / EARTH_METRES_PER_DEGREE,
  };
}

function advance(position, field, seconds) {
  const first = sample(field, position.longitude, position.latitude);
  if (!first) return null;
  const midpoint = displacement(position, first, seconds / 2);
  const middle = sample(field, midpoint.longitude, midpoint.latitude);
  if (!middle) return null;
  return displacement(position, middle, seconds);
}

function advect(seeds, fields) {
  return seeds.map((seed) => {
    let position = { ...seed };
    let stopped = false;
    const points = [{ ...position, stopped }];
    for (const field of fields) {
      if (!stopped) {
        for (let step = 0; step < SUBSTEPS_PER_DAY; step += 1) {
          const next = advance(position, field, DAY_SECONDS / SUBSTEPS_PER_DAY);
          if (!next) {
            stopped = true;
            break;
          }
          position = next;
        }
      }
      points.push({ ...position, stopped });
    }
    return points;
  });
}

function haversineKilometres(a, b) {
  const radians = Math.PI / 180;
  const dLatitude = (b.latitude - a.latitude) * radians;
  const dLongitude = (b.longitude - a.longitude) * radians;
  const latitudeA = a.latitude * radians;
  const latitudeB = b.latitude * radians;
  const value = Math.sin(dLatitude / 2) ** 2
    + Math.cos(latitudeA) * Math.cos(latitudeB) * Math.sin(dLongitude / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
}

self.onmessage = ({ data }) => {
  const trajectories = data.forecasts.map((forecast) => advect(data.seeds, forecast.fields));
  const separation = [];
  if (trajectories.length === 2) {
    for (let lead = 0; lead <= data.maximumLead; lead += 1) {
      let total = 0;
      let count = 0;
      for (let particle = 0; particle < data.seeds.length; particle += 1) {
        const a = trajectories[0][particle][lead];
        const b = trajectories[1][particle][lead];
        if (!a || !b) continue;
        total += haversineKilometres(a, b);
        count += 1;
      }
      separation.push({ lead_day: lead, mean: count ? total / count : 0 });
    }
  }
  self.postMessage({ requestId: data.requestId, trajectories, separation });
};
