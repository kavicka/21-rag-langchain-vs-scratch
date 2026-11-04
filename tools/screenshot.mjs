// Screenshots for the README, taken with Playwright against the running app (make api + make web).
//
//   node tools/screenshot.mjs ui <left pipeline> <right pipeline> "<question>" assets/ui.png
//   node tools/screenshot.mjs phoenix assets/phoenix.png
import { chromium } from "playwright";

const [, , what, ...args] = process.argv;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 2 });

if (what === "ui") {
  const [left, right, question, out] = args;
  await page.goto("http://localhost:5173");
  const selects = page.getByLabel("Pipeline");
  await selects.nth(0).selectOption(left);
  await selects.nth(1).selectOption(right);
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Ask both" }).click();
  // both lanes are done when each shows its "total … ms" badge
  await page.waitForFunction(() => document.querySelectorAll(".badge").length > 0 &&
    [...document.querySelectorAll(".badge")].filter((b) => b.textContent.startsWith("total")).length === 2,
    null, { timeout: 600_000 });
  await page.screenshot({ path: out, fullPage: true });
} else if (what === "phoenix") {
  const [out] = args;
  await page.goto("http://localhost:6006/projects");
  await page.getByText("step21").first().click();
  await page.waitForTimeout(3000);
  // open the slowest LangGraph trace: its tree shows retrieve → grade → rewrite → retrieve → grade → generate
  await page.getByText("LangGraph").first().click();
  await page.waitForTimeout(4000);
  await page.screenshot({ path: out });
}
await browser.close();
