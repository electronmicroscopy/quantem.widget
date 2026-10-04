import * as React from "react";
import { MenuItem, Select } from "@mui/material";
import { DARK_COLORS } from "../theme";
import { sliderStyles } from "../controlStyles";

export const CifControlTheme = React.createContext(DARK_COLORS);

/** Dense themed MUI select, matching the Show3D viewer control family. */
export function CompactSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  options: readonly (readonly [string | number, string])[];
}) {
  const colors = React.useContext(CifControlTheme);
  return (
    <Select
      size="small"
      value={value}
      onChange={(e) => onChange(String(e.target.value))}
      inputProps={{ "aria-label": label }}
      sx={{
        height: 26,
        minWidth: 58,
        fontSize: 12,
        borderRadius: 0,
        color: colors.text,
        backgroundColor: colors.controlBg,
        "& .MuiSelect-select": { py: 0.25, pl: 1, pr: "26px !important" },
        "& .MuiSvgIcon-root": { color: colors.textMuted },
        "& .MuiOutlinedInput-notchedOutline": {
          borderColor: colors.border,
          borderRadius: 0,
        },
        "&:hover .MuiOutlinedInput-notchedOutline": {
          borderColor: colors.accent,
        },
      }}
      MenuProps={{
        transitionDuration: 0,
        MenuListProps: { dense: true },
        PaperProps: {
          sx: {
            backgroundColor: colors.controlBg,
            color: colors.text,
            borderRadius: 0,
            border: `1px solid ${colors.border}`,
          },
        },
      }}
    >
      {options.map(([v, name]) => (
        <MenuItem key={v} value={v} dense>
          {name}
        </MenuItem>
      ))}
    </Select>
  );
}

export const compactSlider = { ...sliderStyles.small, minWidth: 65 };
