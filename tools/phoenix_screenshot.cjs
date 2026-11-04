// Screenshot of one LangGraph trace in Phoenix (run eval/trace.py first): node tools/phoenix_screenshot.cjs assets/phoenix.png
const { chromium } = require("playwright");
(async () => {
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1500, height: 950 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:6006/projects/UHJvamVjdDoy/traces?timeRangeKey=7d");
await p.waitForTimeout(4000);
await p.getByText("LangGraph", { exact: true }).first().click();
await p.waitForTimeout(5000);
console.log(p.url());
await p.screenshot({ path: process.argv[2], clip: { x: 900, y: 45, width: 600, height: 905 } });
await b.close(); })();
