const fs = require('fs');
const glob = require('glob');

const CODELABS_DIR = 'codelabs';
const DEFAULT_PDF_SETTINGS = {
  author: '"@Towhee.io"',
  date: new Date().toLocaleDateString(),
  titlepage: true,
  'titlepage-color': 'fbfbee',
  'titlepage-text-color': 'FFFFFF',
  'titlepage-rule-color': 'FFFFFF',
};
// get all codelab.json from claat generated files
const mdFiles = glob.sync(`${CODELABS_DIR}/*/index.md`);

for (let i = 0; i < mdFiles.length; i++) {
  const mdFile = mdFiles[i];
  const mdFileContents = fs.readFileSync(mdFile, 'utf-8');

  const newMarkdownContent = [];
  const frontmatter = [];
  const lines = mdFileContents.split(/\r?\n/);

  let frontmatter_set = false;

  lines.forEach((line, index) => {
    // remove durations and frontmatter
    if (frontmatter_set) {
      if (!line.startsWith('Duration:')) {
        newMarkdownContent.push(line);
      }
    } else {
      frontmatter.push(line);
      const nextLine = lines[index + 1];
      frontmatter_set = nextLine.startsWith('# ') || nextLine === '';
    }
  });

  // formatOBJ
  const frontmatterObj = array2Obj(frontmatter);
  const newFrontmatter = obj2Array(
    decorateFrontmatter(frontmatterObj, DEFAULT_PDF_SETTINGS)
  );

  // new file name
  const newFile = mdFile.replace(`index.md`, `${frontmatterObj.id}.pdf.md`);

  console.log(`Formatting ${newFile} ...`);
  // save article to disk
  fs.writeFile(
    newFile,
    ['---', ...newFrontmatter, '---', ...newMarkdownContent].join('\n'),
    err => {
      if (err) {
        throw err;
      }
    }
  );
}

function array2Obj(arr) {
  const res = {};
  arr.forEach(item => {
    const [key, value] = item.split(':');
    res[key] = value.trim();
  });
  return res;
}

function decorateFrontmatter(obj, defaultObj) {
  const res = { ...obj };
  for (const key in defaultObj) {
    if (typeof res[key] === 'undefined') {
      res[key] = defaultObj[key];
    }
  }
  if (typeof res.title === 'undefined') {
    res.title = res.summary;
  }
  return res;
}

function obj2Array(obj) {
  const res = [];
  for (const key in obj) {
    res.push(`${key}: ${obj[key]}`);
  }
  return res;
}
