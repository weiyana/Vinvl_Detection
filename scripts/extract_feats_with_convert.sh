PROPOSAL_TYPE="gt"
sh scripts/extract_feats.sh $PROPOSAL_TYPE
cd ../uniter
python data_process/convert_proposal_test.py --proposal $PROPOSAL_TYPE

