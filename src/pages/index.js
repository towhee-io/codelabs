import Layout from '../components/layout';
import { useMemo } from 'react';
import TabList from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import TutorialCard from '../components/card';
import classes from '../../styles/home.module.less';
import ToolBar from '../components/toolBar';
import { useRouter, default as Router } from 'next/router';
import { getCodelabsJson } from '../utils/common';
import * as qs from 'qs';

export default function HomePage({ data = [] }) {
  const { query } = useRouter();
  const { s = '', c = 'all', sort = 'a-z' } = query;

  const handleChange = (e, value) => {
    query.sort = value;
    if (value === '') {
      delete query.sort;
    }
    let q = qs.stringify(query);

    Router.push(`/?${q}`);
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
    query.s = e.target.value;
    if (e.target.value === '') {
      delete query.s;
    }
    let q = qs.stringify(query);

    Router.push(`/?${q}`);
  };

  const handleSelectorChange = e => {
    query.c = e.target.value;
    if (e.target.value === 'all') {
      delete query.c;
    }
    let q = qs.stringify(query);

    Router.push(`/?${q}`);
  };

  const formatData = useMemo(() => {
    let tempData = data.slice();
    if (s) {
      tempData = tempData.filter(({ title }) => title.includes(s));
    }

    if (c !== 'all') {
      tempData = tempData.filter(({ category }) => category.includes(c));
    }
    switch (sort) {
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

      default:
        return tempData;
    }
  }, [data, sort, s, c]);

  return (
    <Layout>
      <section className={classes.homeContainer}>
        <Box className={classes.welcome}>
          <Box className={classes.inner}>
            <Typography component="h2">Welcome to Towhee Codelabs!</Typography>
            <Box>
              <Typography component="p">
                Towhee Codelabs provide a guided, tutorial, hands-on towhee
                integration experience. Most tutorials will step you through the
                process of the unstructured data, such as reverse image search,
                reverse video search, audio classification, question and answer
                systems, molecular search, etc. Most of the bootcamp can be
                found at https://github.com/towhee-io/examples.
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
                value={sort}
                onChange={handleChange}
                className={classes.tabBar}
              >
                <Tab label="A-Z" value="a-z" />
                <Tab label="RECENT" value="recent" />
                <Tab label="DURATION" value="duration" />
              </TabList>

              <ToolBar
                keyWord={s}
                handleKeyWordChange={handleSearchChange}
                categoryVal={c}
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
