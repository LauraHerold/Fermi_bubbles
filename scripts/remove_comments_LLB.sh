# remove comments
cd /Users/Dmitry/Work/student_works/github_bubbles/Fermi_bubbles_34670
out=/Users/Dmitry/Work/student_works/github_bubbles/Fermi_bubbles_34670
for fn in 0gp_bubbles_head
do
sed -ie 's/\\%/ReallyCool/g' $out/$fn.tex
sed -ie 's/%.*$/%/' $out/$fn.tex
sed -ie 's/ %//' $out/$fn.tex
sed -ie '/^%$/d' $out/$fn.tex
sed -ie 's/ReallyCool/\\%/g' $out/$fn.tex
done
