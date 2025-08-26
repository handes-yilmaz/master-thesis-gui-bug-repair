#!/usr/bin/env node
import { chromium } from 'playwright'
import { analyze } from '@axe-core/playwright'

const args = process.argv.slice(2)
const urlIdx = args.indexOf('--url')
const url = urlIdx >= 0 ? args[urlIdx+1] : 'http://localhost:5173'

const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: 1280, height: 800 }})
const page = await context.newPage()
await page.goto(url)

const results = await analyze(page, { detailedReport: true, detailedReportOptions: { html: false } })
console.log(JSON.stringify(results.violations, null, 2))

await browser.close()

if (results.violations.length > 0) {
  process.exitCode = 2
}
