a = 11
b = 2.2
c = a/b
print(c)

a = 2
b = 3
a, *b = [1,2,3]
print(a,b)
a, *b, c = [1,2,3,4]
print(a,b, c)

str1 = "abcd"
str2 = "abcd"

s1 = "praveen"
s2=''
for i in range(1, len(s1)+1):
    s2 +=s1[len(s1)-i]

print(s2)
