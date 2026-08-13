./runsmallmodelsa.sh &
./runsmallmodelsb.sh &
./runsmallmodelsc.sh &

# wait for all to complete
wait
echo "All small-model scripts have finished."