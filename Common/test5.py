# coding: utf-8
for chunk in a:
    i = 0
    l = []
    while i < len(offsets):
        l.append({names[i]: np.frombuffer(chunk[offsets[i]: int(offsets[i] + types[i](1).itemsize)], dtype=types[i])[0]})
        i += 1
    p.append(l)
    
