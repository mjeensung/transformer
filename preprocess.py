import xml.etree.ElementTree as ET

doc = ET.parse("./dataset/de-en/IWSLT16.TED.tst2010.de-en.de.xml")

root = doc.getroot()

with open('./dataset/de-en/test.de',"w") as f:
    for seg in root.iter("seg"):
        f.write(seg.text+"\n")

doc = ET.parse("./dataset/de-en/IWSLT16.TED.tst2010.de-en.en.xml")

root = doc.getroot()

with open('./dataset/de-en/test.en',"w") as f:
    for seg in root.iter("seg"):
        f.write(seg.text +"\n")
