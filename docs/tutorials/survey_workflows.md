# Survey workflows

Use these workflows when you are reviewing a full microscopy session instead of
opening one dataset at a time.

- [Folder survey](./survey): start here for a fast first pass. Point
  `survey(folder)` at a folder and review HAADF/STEM thumbnails, EDS launchers,
  metadata, and stars/checkmarks in one place.
- [Field-of-view survey](./survey_fov): use this when the session has repeated
  acquisitions of the same region, such as 0 degree, 90 degree, and EDS files.
  Related files are grouped into compact rows so you can choose the best regions
  for follow-up analysis.

Both workflows save selection state next to the data as `.quantem-survey.json`,
so you can curate a session without deleting, renaming, or moving raw files.

After selecting useful files, continue to [Saving and sharing](./widget_export)
to make notebooks or HTML files portable.
