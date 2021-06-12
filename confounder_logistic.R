## p11 = baseline conversion in absence of confounder, p12 = baseline conversion in presence of confounder
## pz = confounder proportion in population, px = proportion of sample receiving exposure to x
## p2 = minimum detectable conversion ration (after exposure x)

func1 = function(alpha, powr, p11, p12, pz, p2, px) {
  A = p11/(1-p11)
  G = (p12/(1-p12)) / (p11/(1-p11))
  C = px/(1-px)
  D = 1
  
  p11_prime = p11/(1-p11)
  p12_prime = p12/(1-p12)
  qz = 1 - pz
  a = (1-p2)*p11_prime*p12_prime
  b = p11_prime*(qz - p2) + p12_prime*(pz - p2)
  c = -p2
  
  plus = (-b+sqrt(b^2 - 4*a*c))/(2*a)
  minus = (-b-sqrt(b^2 - 4*a*c))/(2*a)
  
  mylist = list("plus" = plus, "minus" = minus)
  return(mylist)
}
