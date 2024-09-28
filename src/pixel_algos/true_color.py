import math

maxR = 3.0  # max reflectance
midR = 0.13
sat = 1.2
gamma = 1.8

gOff = 0.01
gOffPow = math.pow(gOff, gamma)
gOffRange = math.pow(1.0 + gOff, gamma) - gOffPow

mode_depth: dict[str, float] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "int16": 32767.0
}

def evaluatePixelArray(smp, depth_in=65535.0, depth_out=256.0):
    rgbLin = satEnh(sAdj(smp[0] / depth_in), sAdj(smp[1] / depth_in), sAdj(smp[2] / depth_in))

    return [sRGB(rgbLin[0]) * depth_out, sRGB(rgbLin[1]) * depth_out, sRGB(rgbLin[2]) * depth_out]


def clip(s):
    if s > 1.0:
        return 1.0
    elif s < 0.0:
        return 0.0
    else:
        return s


def adj(a, tx, ty, maxC):
    ar = clip(a / maxC)
    return ar * (ar * (tx / maxC + ty - 1.0) - ty) / (ar * (2.0 * tx / maxC - 1.0) - tx / maxC)


def adjGamma(b):
    return (math.pow(b + gOff, gamma) - gOffPow) / gOffRange


def sAdj(a):
    return adjGamma(adj(a, midR, 1.0, maxR))


def sRGB(c):
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * math.pow(c, 0.41666666666) - 0.055


def satEnh(r, g, b):
    avgS = (r + g + b) / 3.0 * (1.0 - sat)
    return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]
