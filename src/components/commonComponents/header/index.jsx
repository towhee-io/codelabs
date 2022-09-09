import Link from '@mui/material/Link';
import { NAV_LIST } from './constants';
import MenuIcon from '@mui/icons-material/Menu';
import { useState, useRef, useEffect } from 'react';
import classes from './index.module.less';
import clsx from 'clsx';
import { TohiLogo } from '../footer/icons';
import { getGithubStatis } from '../http';
import GitHubButton from '../githubButton';

const Header = () => {
  const content = useRef(null);

  const [openMask, setOPenMask] = useState(false);
  const [stat, setStat] = useState({ star: 0, forks: 0 });

  const handleToggleMobileMenu = () => {
    setOPenMask(v => !v);
  };

  const handleClickOutside = e => {
    if (e.target.contains(content.current) && e.target !== content.current) {
      setOPenMask(false);
    }
  };

  const NavSection = ({ navList, styles, isDeskTop = true }) => (
    <ul
      className={clsx(styles.navList, {
        [styles.desktopNav]: isDeskTop,
      })}
    >
      <li key="github" className={classes.gitBtnsWrapper}>
        <GitHubButton
          stat={stat}
          type="star"
          href="https://github.com/towhee-io/towhee"
        >
          Star
        </GitHubButton>

        <GitHubButton
          stat={stat}
          type="fork"
          href="https://github.com/towhee-io/towhee"
        >
          Forks
        </GitHubButton>
      </li>
      {navList.map(v => (
        <li key={v.label}>
          <Link href={v.href} underline="none">
            {v.label}
          </Link>
        </li>
      ))}
    </ul>
  );

  useEffect(() => {
    (async function getData() {
      try {
        const { stargazers_count, forks_count } = await getGithubStatis();
        setStat({ star: stargazers_count, forks: forks_count });
      } catch (error) {
        console.log(error);
      }
    })();
  }, []);

  return (
    <section className={classes.headerWrapper}>
      <div className={classes.headerContent}>
        <div className={classes.leftPart}>
          <Link href="https://towhee.io/" className={classes.linkBtn}>
            <TohiLogo />
          </Link>
        </div>
        <div className={classes.rightPart}>
          <NavSection navList={NAV_LIST} styles={classes} />
          <div className={classes.loginBtn} key="signIn">
            <a href="https://towhee.io/user/login">Sign In</a>
          </div>
          <div className={classes.menuWrapper} onClick={handleToggleMobileMenu}>
            <MenuIcon />
          </div>
        </div>
        <div
          className={clsx(classes.menuMask, {
            [classes.active]: openMask,
          })}
          onClick={handleClickOutside}
        >
          <div className={classes.menuContent} ref={content}>
            <NavSection navList={NAV_LIST} styles={classes} isDeskTop={false} />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Header;
