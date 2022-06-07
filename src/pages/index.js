import Layout from '../components/layout';
import { useMemo, useState } from 'react';
import TabList from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import TutorialCard from '../components/card';
import classes from '../../styles/home.module.less';
import ToolBar from '../components/toolBar';
import { useRouter } from 'next/router';
import { getCodelabsJson } from '../utils/common';

export default function HomePage({ data = [] }) {
  const { pathname } = useRouter();
  const [keyWord, setKeyWord] = useState('');
  const [categoryVal, setCategoryVal] = useState('all');
  const [sortWay, setSortWay] = useState('a-z');

  const handleChange = (e, value) => {
    setSortWay(value);
  };

  const categoryOptions = useMemo(() => {
    const options = [];
    data.forEach(({ category }) => {
      category.forEach(v => {
        if (options.includes(v)) return;
        options.push(v);
      });
    });

    return [
      {
        value: 'all',
        label: 'All',
        id: 0,
      },
      ...options.map((v, i) => ({
        label: v,
        value: v,
        id: i,
      })),
    ];
  }, [data]);

  const handleSearchChange = e => {
    setKeyWord(e.target.value);
  };

  const handleSelectorChange = e => {
    setCategoryVal(e.target.value);
  };

  const formatData = useMemo(() => {
    let tempData = data.slice();
    if (keyWord) {
      tempData = tempData.filter(({ title }) => title.includes(keyWord));
    }

    if (categoryVal !== 'all') {
      tempData = tempData.filter(({ category }) =>
        category.includes(categoryVal)
      );
    }
    switch (sortWay) {
      case 'a-z':
        return tempData.sort(
          (x, y) => x.title.charCodeAt(0) - y.title.charCodeAt(0)
        );
      case 'recent':
        return tempData.sort(
          (x, y) =>
            new Date(x.updated).getTime() - new Date(y.updated).getTime()
        );
      case 'duration':
        return tempData.sort((x, y) => x.duration - y.duration);
    }
  }, [data, sortWay, pathname, keyWord, categoryVal]);

  return (
    <Layout>
      <section className={classes.homeContainer}>
        <Box className={classes.welcome}>
          <Box className={classes.inner}>
            <Typography component="h2">Welcome to Towhee Codelabs!</Typography>
            <Box>
              <Typography component="p">
                Milvus Codelabs provide a guided, tutorial, hands-on towhee
                integration experience. Most tutorials will step you through the
                process of installation, building a milvus application, or
                integrate towhee with your existing application.
              </Typography>
            </Box>
          </Box>
        </Box>
        <Box>
          <Box className={classes.inner}>
            <Stack
              direction="row"
              justifyContent="space-between"
              alignItems="center"
            >
              <TabList
                value={sortWay}
                onChange={handleChange}
                className={classes.tabBar}
              >
                <Tab label="A-Z" value="a-z" />
                <Tab label="RECENT" value="recent" />
                <Tab label="DURATION" value="duration" />
              </TabList>

              <ToolBar
                keyWord={keyWord}
                handleKeyWordChange={handleSearchChange}
                categoryVal={categoryVal}
                handleSelectorChange={handleSelectorChange}
                options={categoryOptions}
              />
            </Stack>

            <Box className={classes.cardLayout}>
              {formatData.map(v => (
                <TutorialCard {...v} key={v.id} />
              ))}
            </Box>
          </Box>
        </Box>
      </section>
    </Layout>
  );
}

export const getStaticProps = async () => {
  // const res = await axiosInstance.get('/codelabs');
  const data = getCodelabsJson();

  return {
    props: {
      data,
    },
  };
};
