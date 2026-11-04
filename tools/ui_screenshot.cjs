// Screenshot of the two-lane UI for the README. Needs `make api` (port 8000) and `make web` (port 5173) running.
//   node tools/ui_screenshot.cjs <left pipeline> <right pipeline> "<question>" assets/ui.png
// CommonJS on purpose: a globally installed playwright resolves through NODE_PATH only for require().
const { chromium } = require("playwright");

(async () => {
  const [left, right, question, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 2 });
  await page.goto("http://localhost:5173");
  const selects = page.getByLabel("Pipeline");
  await selects.nth(0).selectOption(left);
  await selects.nth(1).selectOption(right);
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Ask both" }).click();
  // both lanes are done when each shows its "total … ms" badge
  await page.waitForFunction(
    () => [...document.querySelectorAll(".badge")].filter((b) => b.textContent.startsWith("total")).length === 2,
    null,
    { timeout: 900_000 },
  );
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
})();
