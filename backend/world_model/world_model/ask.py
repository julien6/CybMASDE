continue_refinement = input("Continue to the next refinement cycle? (Y/N): ").strip()

f = open("result.txt", "w")
f.write(continue_refinement)
f.close()
