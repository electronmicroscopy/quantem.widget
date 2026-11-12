import { afterEach, describe, expect, it } from "vitest";
import { standaloneWidgetStaticHtmlFromDocument } from "./format";

// The page export_html writes: gzipped offline runtime, its loader, widget state and view.
const EXPORTED_BODY = `
<script type="application/gzip" id="quantem-widget-runtime-0">AAAA</script>
<script type="application/gzip" id="quantem-widget-runtime-1">BBBB</script>
<script id="quantem-widget-runtime-loader">/* loader */</script>
<script type="application/vnd.jupyter.widget-state+json">{"state":{}}</script>
<script type="application/vnd.jupyter.widget-view+json">{"model_id":"m1"}</script>
<div class="rendered-widget">drawn by the runtime</div>
<script>/* injected at run time */</script>`;

describe("standaloneWidgetStaticHtmlFromDocument", () => {
  afterEach(() => { document.body.innerHTML = ""; });

  it("keeps the inline offline runtime so a re-saved page still opens", () => {
    document.body.innerHTML = EXPORTED_BODY;
    const html = standaloneWidgetStaticHtmlFromDocument();
    for (const id of ["quantem-widget-runtime-0", "quantem-widget-runtime-1", "quantem-widget-runtime-loader"]) {
      expect(html).toContain(`id="${id}"`);
    }
    expect(html).toContain("application/vnd.jupyter.widget-state+json");
    expect(html).toContain("application/vnd.jupyter.widget-view+json");
    expect(html).not.toContain("drawn by the runtime");
    expect(html).not.toContain("injected at run time");
  });
});
