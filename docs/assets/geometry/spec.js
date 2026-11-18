/* Geometry for display only. This module never changes or executes the scientific model. */
(function (factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else window.CarbonGeometry = factory();
})(function () {
  'use strict';
  const positive = (value, name) => {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Missing or invalid geometry: ${name}`);
    return value;
  };
  function fromSource(source) {
    const mass = positive(source.common?.dry_composite_mass_kg, 'dry composite mass');
    const pellet = source.cases?.find(row => row.case_id === 'pellet_24mm_electrical_precision')?.geometry;
    const channel = source.cases?.find(row => row.case_id === 'channel_500um_electrical_precision')?.geometry;
    if (!pellet || !channel) throw new Error('The saved pilot geometry is unavailable.');
    for (const [name, value] of Object.entries(pellet)) positive(value, name);
    for (const [name, value] of Object.entries(channel)) positive(value, name);
    if (!Number.isInteger(pellet.parallel_beds) || !Number.isInteger(channel.parallel_units) || pellet.parallel_beds > 40 || channel.parallel_units > 200) throw new Error('Geometry exceeds the bounded viewer size.');
    const width = mass / (pellet.bulk_density_kg_m3 * pellet.parallel_beds * pellet.bed_thickness_m * pellet.path_length_m);
    const channelWidth = mass / (2 * channel.coating_density_kg_m3 * channel.coating_thickness_m * channel.parallel_units * channel.path_length_m);
    const channelPitch = 2 * channel.coating_thickness_m + channel.support_thickness_m + channel.channel_gap_m;
    if (!Number.isFinite(width) || !Number.isFinite(channelWidth) || width > 5 || channelWidth > 10 || pellet.bed_thickness_m > 1 || channel.parallel_units * channelPitch > 3) throw new Error('Geometry is outside the display envelope.');
    if (pellet.pellet_diameter_m * 4 > pellet.bed_thickness_m) throw new Error('Enlarged sample particles do not fit this bed.');
    const enlargedStep = pellet.pellet_diameter_m * 4 * 1.11;
    if (width / enlargedStep * pellet.path_length_m / enlargedStep * pellet.bed_thickness_m / enlargedStep > 20000) throw new Error('Particle sample exceeds the display budget.');
    return {
      mass, pellet: {...pellet, width_m: width, mass_per_bed_kg: mass / pellet.parallel_beds},
      channel: {...channel, width_m: channelWidth, pitch_m: channelPitch,
        stack_height_m: channel.parallel_units * channelPitch},
      display: {pellet_enlargement: 4, bank_spacing_m: 0.012, frame_margin_m: 0.006,
        note: 'Particle sampling, frame, seals, fasteners and bank spacing are illustrative.'}
    };
  }
  return {fromSource};
});
