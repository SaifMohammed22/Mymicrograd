class F_Value:
    def __init__(self, val, dot=0.0, label=''):
        self.val = val
        self.dot = dot
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, F_Value) else F_Value(other, 0.0)
        return F_Value(self.val + other.val, self.dot + other.dot)
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        # f = a * b -> df/da = da/da * b , df/db = db/db * a
        other = other if isinstance(other, F_Value) else F_Value(other, 0.0)
        return F_Value(self.val * other.val, self.dot * other.val + self.val * other.dot)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        return F_Value(self.val ** other, other * (self.val ** (other - 1)) * self.dot)

    def relu(self):
        if self.val > 0:
            return F_Value(self.val, self.dot)
        else:
            return F_Value(0.0, 0.0)

    def __neg__(self):
        return F_Value(-self.val, -self.dot)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, F_Value) else F_Value(other, 0.0)
        return F_Value(self.val / other.val, (self.dot * other.val - self.val * other.dot) / (other.val ** 2))

    def __repr__(self):
        return f"Dual(val={self.val}, dot={self.dot})"
    
if __name__ == "__main__":
    x = F_Value(2.0, dot=1.0, label='x')   # seed derivative wrt x
    y = F_Value(3.0, dot=0.0, label='y')
    f = x * y + x ** 2
    print(f.val, f.dot) 
    x = F_Value(2.0, dot=0.0, label='x')
    y = F_Value(3.0, dot=1.0, label='y')   # seed derivative wrt y
    f = x * y + x ** 2
    print(f.val, f.dot)