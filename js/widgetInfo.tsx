import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export type MetadataRow = [label: string, value: React.ReactNode];

export function MetadataTable({ rows }: { rows: MetadataRow[] }) {
  const visibleRows = rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (visibleRows.length === 0) return null;
  return (
    <Box
      component="table"
      sx={{
        borderCollapse: "collapse",
        "& td": { py: 0.2, fontSize: 11, lineHeight: 1.35, verticalAlign: "top" },
        "& td:first-of-type": { pr: 1.25, opacity: 0.7, whiteSpace: "nowrap" },
        "& td:last-of-type": { fontFamily: "monospace" },
      }}
    >
      <tbody>
        {visibleRows.map(([label, value]) => (
          <tr key={label}>
            <td>{label}</td>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </Box>
  );
}

export function MetadataSection({ rows }: { rows: MetadataRow[] }) {
  if (rows.filter(([, value]) => value !== null && value !== undefined && value !== "").length === 0) return null;
  return (
    <>
      <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Data</Typography>
      <MetadataTable rows={rows} />
    </>
  );
}
