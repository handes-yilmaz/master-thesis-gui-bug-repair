#!/usr/bin/env node
import fs from 'node:fs'
import { PNG } from 'pngjs'
import pixelmatch from 'pixelmatch'

const [,, beforePath, afterPath, outPath] = process.argv
if (!beforePath || !afterPath || !outPath){
  console.error('Usage: node visual-diff.mjs before.png after.png diff.png')
  process.exit(1)
}

const img1 = PNG.sync.read(fs.readFileSync(beforePath))
const img2 = PNG.sync.read(fs.readFileSync(afterPath))

const {width, height} = img1
const diff = new PNG({width, height})

const mismatched = pixelmatch(img1.data, img2.data, diff.data, width, height, { threshold: 0.1 })
fs.writeFileSync(outPath, PNG.sync.write(diff))
console.log(JSON.stringify({ width, height, mismatched }))
