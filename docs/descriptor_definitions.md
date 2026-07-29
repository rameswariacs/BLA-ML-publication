# Descriptor Definitions

The model uses 19 geometry-derived descriptors calculated from Cartesian coordinates.

## Descriptor Families

### Global pi-surface and overlap

- `projected_pi_overlap_area`: projected overlap area, `S_ov`.
- `projected_pi_overlap_fraction_min`: minimum projected overlap fraction, `f_ov,min`.
- `pi_projected_area_mean`: mean projected pi-area, `Sbar_pi`.
- `pi_projected_area_ratio_min_over_max`: projected-area ratio, `R_S`.
- `delta_overlap_fraction`: overlap-fraction difference, `Delta f_ov`.

### Stacking geometry

- `stack_interplane_distance`: interplanar distance, `d_perp`.
- `stack_lateral_slip`: lateral slip, `d_slip`.
- `stack_plane_normal_angle_deg`: plane-normal angle, `theta_plane`.

### Local interfragment C...C contact network

- `contact_min_cc_distance`: minimum interfragment C...C distance, `d_min`.
- `mean_top10_closest_c_distances`: mean of ten shortest interfragment C...C distances, `d10`.
- `contact_interfragment_distance_std`: spread of interfragment C...C distances, `sigma_C...C`.
- `n_interfragment_c_contacts_3p4`: number of interfragment C...C pairs within 3.4 A, `N_3.4`.
- `n_interfragment_c_contacts_3p6`: number of interfragment C...C pairs within 3.6 A, `N_3.6`.
- `n_interfragment_c_contacts_4p0`: number of interfragment C...C pairs within 4.0 A, `N_4.0`.
- `contact_density_3p6_per_overlap_area`: 3.6 A contact density normalized by projected overlap area.
- `delta_contact_atoms_3p4`: difference in unique contact atoms within 3.4 A.

### Bond-length alternation and fragment asymmetry

- `bla_fragment_mean`: mean fragment BLA.
- `contact_region_bla_mean`: mean contact-region BLA.
- `delta_contact_region_bla`: difference in contact-region BLA between fragments.

Full equations and chemical justifications are provided in the Supporting Information.
