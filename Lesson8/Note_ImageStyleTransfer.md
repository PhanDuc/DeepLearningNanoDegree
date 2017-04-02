# Image Style Transfer

Use `VGG-16`: Content image + Style image --> New Content + Style

Loss Function: Content Loss + Style Loss

- Content Loss: use VGG high-level to recognize contents 
- Style Loss: messure feature map correlation -- filter/weight similarity
- Use loss to update `IMAGE`, not the `weight`

## Additional Reading

### Paper

- [Original Paper](https://arxiv.org/abs/1508.06576)

### Youtube-corresponding tutortial

- [Github Content](https://github.com/llSourcell/How-to-Generate-Art-Demo/blob/master/demo.ipynb)

### Blog Explanation

- [From Harish Narayanan](https://harishnarayanan.org/writing/artistic-style-transfer/)
- [From ml4a](https://ml4a.github.io/ml4a/style_transfer/)
- [From Genekogan](http://genekogan.com/works/style-transfer/)
- [From Julia Evans](https://jvns.ca/blog/2017/02/12/neural-style/)

### Apps

- [pikaoapp](http://www.pikazoapp.com/)
- [Deep Art](https://deepart.io/)
- [Artisto](https://artisto.my.com/)
- [Prisma](https://prisma-ai.com/)