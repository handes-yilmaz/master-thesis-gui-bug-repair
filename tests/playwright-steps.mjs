#!/usr/bin/env node
import { chromium } from 'playwright'
import fs from 'node:fs'

function parseArg(flag, fallback){
  const i = process.argv.indexOf(flag)
  return i>=0 ? process.argv[i+1] : fallback
}

const url = parseArg('--url', 'http://localhost:5173')
const stepsPath = parseArg('--steps', 'scripts/steps.example.json')
const outDir = parseArg('--out', 'runs/run1')

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive:true})

const steps = JSON.parse(fs.readFileSync(stepsPath,'utf-8'))
const before = steps.before_wait_ms ?? 400
const after = steps.after_wait_ms ?? 800
const vp = steps.viewport ?? { width: 1280, height: 800 }

const browser = await chromium.launch()
const context = await browser.newContext({ viewport: vp })
const page = await context.newPage()

await page.goto(url)
await page.waitForTimeout(before)
await page.screenshot({ path: `${outDir}/before.png`})

for (const act of (steps.actions || [])) {
  if (act.type === 'click') await page.click(act.selector)
  else if (act.type === 'fill') await page.fill(act.selector, act.value || '')
  else if (act.type === 'press') await page.keyboard.press(act.key)
  else if (act.type === 'wait') await page.waitForTimeout((act.seconds || 0.5)*1000)
}

await page.waitForTimeout(after)
await page.screenshot({ path: `${outDir}/after.png`})

await browser.close()
console.log('Saved before/after screenshots to', outDir)
