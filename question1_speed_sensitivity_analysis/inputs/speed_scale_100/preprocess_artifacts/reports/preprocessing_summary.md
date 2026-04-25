# Preprocessing Summary

## Inputs
- Coordinate file: `客户坐标信息.xlsx`
- Time-window file: `时间窗.xlsx`
- Orders file: `订单信息.xlsx`
- Distance file: `距离矩阵.xlsx`

## Data quality snapshot
- Order rows: 2169
- Missing weight rows repaired: 14
- Missing volume rows repaired: 16
- Total imputed rows: 30
- Oversize orders: 2
- Active customers with orders: 88
- Zero-demand customers: 10
- Customers requiring split service: 37

## Green-zone note
- Geometry-based count within 10 km of (0,0): 15
- Problem statement states 30 customers in the green zone.
- This implementation follows the attachment geometry and flags the discrepancy for reporting.

## Policy preview
- Green-zone customers that are EV-only under the ban: 9
- Green-zone customers that fuel vehicles may serve after 16:00: 6
- Mandatory-EV active customers: 8
- Mandatory-EV total weight: 30149.462 kg
- Mandatory-EV total volume: 84.006 m^3
- Optimistic lower bound on EV trips by weight: 11
- Optimistic lower bound on EV trips by volume: 6

## Outputs
- Clean tables are stored under `tables/` in CSV and Excel workbook form.
- Minute-level travel time lookup is stored in `travel_time_lookup.npz`.
- Visualizations are stored under `figures/`.
