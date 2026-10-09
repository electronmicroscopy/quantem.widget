import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

type MetadataRow = [label: string, value: React.ReactNode];

/** Rows with a value to show: null, undefined and empty strings are dropped. */
function visibleMetadataRows(rows: MetadataRow[]): MetadataRow[] {
  return rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
}

function MetadataTable({ rows }: { rows: MetadataRow[] }) {
  const visibleRows = visibleMetadataRows(rows);
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
  if (visibleMetadataRows(rows).length === 0) return null;
  return (
    <>
      <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Data</Typography>
      <MetadataTable rows={rows} />
    </>
  );
}
