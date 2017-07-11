import barebones_enc_dec as bed

predictor=bed.getModel()
answer=bed.query("play","software",predictor)
print "Answer:",answer
answer=bed.query("bad","advice",predictor)
print "Answer:",answer
answer=bed.query("gangal","moron",predictor)
print "Answer:",answer
answer=bed.query("great","expectations",predictor)
print "Answer:",answer
