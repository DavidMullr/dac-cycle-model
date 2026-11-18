// Direction guides for the adsorption view. Coordinates follow the displayed geometry.
// There is no velocity, pressure, concentration or pore-scale flow calculation here.
export function airflowLayout(specification, state, unit = 10) {
  const p = specification.pellet, c = specification.channel;
  const lanes = [];
  const length = (state.model === 'channel' ? c.path_length_m : p.path_length_m) * unit;
  const add = (x, y, clearance) => lanes.push({x, y, clearance, faceIn: length / 2, faceOut: -length / 2,
    inlet: length / 2 + 1.05, outlet: -length / 2 - 1.05});
  if (state.model === 'channel') {
    const n = state.cutaway ? Math.min(24, c.parallel_units) : c.parallel_units;
    const pitch = c.pitch_m * unit * (c.parallel_units / n) * (state.explode ? 1.7 : 1);
    const sheet = (2 * c.coating_thickness_m + c.support_thickness_m) * unit;
    // Sample passages inside the stack, including their display-only expanded spacing.
    const indices = [...new Set([0, .23, .46, .69, .92].map(f => Math.floor(f * (n - 1))))];
    for (const i of indices) {
      const y = .06 + i * pitch + sheet + (pitch - sheet) / 2;
      for (const f of [-.30, 0, .30]) add(c.width_m * unit * f, y, pitch - sheet);
    }
  } else {
    const count = state.model === 'bank' ? p.parallel_beds : 1;
    const thickness = (p.bed_thickness_m + 2 * p.face_plate_thickness_m) * unit;
    const spacing = specification.display.bank_spacing_m * unit * (state.explode ? 2.8 : 1);
    for (let i = 0; i < count; i++) {
      const base = state.model === 'bank' ? .10 + i * (thickness + spacing) : 0;
      const shift = state.model === 'bank' && state.explode ? (i % 2 ? 1 : -1) * .10 : 0;
      const y = base + (p.face_plate_thickness_m + p.bed_thickness_m / 2) * unit;
      for (const f of (state.model === 'bank' ? [-.24, .24] : [-.34, -.17, 0, .17, .34])) {
        add(p.width_m * unit * f + shift, y, p.bed_thickness_m * unit);
      }
    }
  }
  return {lanes, direction: [0, 0, -1], phase: 'adsorption', quantitative: false};
}

export function airflowPosition(lane, progress) {
  const fraction = ((progress % 1) + 1) % 1;
  const z = lane.inlet + (lane.outlet - lane.inlet) * fraction;
  return {x: lane.x, y: lane.y, z, region: z > lane.faceIn ? 'inlet' : z < lane.faceOut ? 'outlet' : 'bed'};
}

export function airflowMarkerRadius(lane, z) {
  // Keep markers within narrow passages, including the shaft crossing a mouth.
  return z < lane.faceIn + .06 && z > lane.faceOut - .22 ? Math.min(.033, lane.clearance * .4) : .033;
}
