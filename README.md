# GLiNER Sports NER

## TL;DR

## Benchmark Results

## Live Demo

## Architecture

## Dataset
- 161 articles scraped across NBA, NFL, MLB, NHL
- Source: ESPN internal JSON API
- 3,967 annotated training examples after validation
- 22,892 total entity spans across 10 entity types
- 5.77 avg entities per example
- 0.1% flagged/removed by quality filter
- 93.9% weak label coverage vs ESPN metadata
- Splits: train=3,173 / val=396 / test=398 (80/10/10)
- Label distribution: PLAYER > TEAM > STAT >> VENUE, AWARD (class imbalance noted)

## Fine-Tuning Approach

## Evaluation Framework

## Local Setup

## API Reference

## Key Technical Decisions

## Acknowledgments
