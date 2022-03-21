# Implicit User Awareness Modeling via Candidate Items for CTR Prediction in Search Ads

This repo is the official implementation for the WWW 2022 paper: *Implicit User Awareness Modeling via Candidate Items for CTR Prediction in Search Ads*.

## Data format

In the data files, each row corresponds to a search session. Each column in the data represents userID, queryID, label list, target items and request item queue. Each item is consist of itemID, categoryID, brandID, vendorID and priceID. 
All data have been desensitized.

## Quick start

Create a new `data` folder and put the downloaded dataset into the folder. Then,

```bash
python src/main.py 
```
