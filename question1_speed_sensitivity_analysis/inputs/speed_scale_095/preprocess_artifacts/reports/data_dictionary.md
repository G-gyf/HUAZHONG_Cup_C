# Output Data Dictionary

## Tables
- `node_master`: all 99 nodes with coordinates, node type, and geometry-based green-zone label.
- `time_windows_numeric`: customer time windows in both raw HH:MM and minute form.
- `orders_clean`: cleaned order-level demand with imputation flags and oversize markers.
- `customer_demand_98`: 98-customer demand table including zero-demand customers.
- `active_customer_demand_88`: subset of customers with at least one order.
- `customer_master_98`: merged customer coordinates, time windows, demand, and policy feasibility flags.
- `policy_feasibility`: policy-focused customer summary for the green-zone restriction analysis.
- `speed_profile`: piecewise deterministic speed segments used for travel-time lookup.
- `distance_matrix_clean`: cleaned 99x99 road distance matrix indexed by node ID.
- `oversize_order`: order-level records above the largest single-vehicle capacity.
- `ev_policy_summary`: one-row lower-bound summary for mandatory-EV demand under the policy.
- `route_solution_template`: empty schema for later route-level post-solution analysis.

## Key fields
- `tw_start_min` / `tw_end_min`: minutes after 08:00.
- `in_green_zone`: geometry-based label using radius <= 10 km around (0,0).
- `customer_split_required`: customer demand exceeds 3000 kg or 15 m^3.
- `must_use_ev_under_policy`: green-zone customer whose full time window is inside [08:00, 16:00].
- `fuel_service_window_start_min` / `fuel_service_window_end_min`: feasible service interval for fuel vehicles after the ban ends.

## Travel-time lookup file
- `travel_time_lookup.npz` contains arrays `travel_time_minutes`, `departure_minutes`, and `node_ids`.
- `travel_time_minutes[k, i, j]` is the travel time in minutes from node `i` to node `j` when departing at `departure_minutes[k]`.
- Values become `NaN` if the trip cannot be completed before 21:00 under the chosen speed profile.
