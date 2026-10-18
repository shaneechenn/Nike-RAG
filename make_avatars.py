"""Generate the custom chat avatars (run once, or whenever you want to change them)."""
import math

from PIL import Image, ImageDraw

# User avatar: Instagram-style default (gray circle + white person silhouette)
u = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
d = ImageDraw.Draw(u)
d.ellipse([0, 0, 96, 96], fill="#d5d5d5")   # background circle
d.ellipse([30, 22, 66, 58], fill="#ffffff")  # head
d.ellipse([8, 60, 88, 100], fill="#ffffff")  # shoulders
u.save("user_avatar.png")

# Assistant avatar: dark gray circle + white 4-point sparkle (AI mark)
a = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
d2 = ImageDraw.Draw(a)
d2.ellipse([0, 0, 96, 96], fill="#3a3a3a")
pts = []
for i in range(8):
    ang = math.pi * i / 4 - math.pi / 2
    r = 30 if i % 2 == 0 else 11
    pts.append((48 + r * math.cos(ang), 48 + r * math.sin(ang)))
d2.polygon(pts, fill="#ffffff")
a.save("assistant_avatar.png")

print("Saved user_avatar.png and assistant_avatar.png")
