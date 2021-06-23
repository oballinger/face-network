### Turn unstructured image data into a social network

Given a large volume of images of people this tool:

1. Extracts faces 
2. Clusters faces based on similarity 
3. Creates a social network graph based on co-appearance in images 

The three steps above correspond to the black arrows on the left of the diagram below: 


## 1. Extracting faces 
```python
face_network.extract_faces(source_dir, age_gender=False)
```

This function extracts all faces from a directory of images using Dlib’s face detector, and must be run prior to further analysis. 



You can use the [editor on GitHub](https://github.com/oballinger/face-network/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/oballinger/face-network/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
