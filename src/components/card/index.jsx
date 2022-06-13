import Link from 'next/Link';
import { format } from 'date-fns';
import classes from './index.module.less';

export default function TutorialCard(props) {
  const { title, duration, summary, authors, mainCategory, updated, url, id } =
    props;

  const pdfLink = `https://github.com/towhee-io/codelabs/raw/master/pdf/${id}.pdf`;

  return (
    <div className={classes.tutorialCardWrapper}>
      <div className={classes.topContainer}>
        <span className={classes.author}>{authors}</span>
        <span className={classes.time}>
          {format(new Date(updated), 'LLL dd, yyyy')}
        </span>
      </div>
      <h2>
        <Link
          href={`/${url}`}
          target="_self"
          className={classes.title}
          passHref
        >
          <a>{title}</a>
        </Link>
      </h2>

      <p>{summary}</p>
      <Link href={`/${url}`} target="_self">
        <a className={classes.linkButton}>Start</a>
      </Link>
      <div className={classes.bottomContainer}>
        <span className={classes.tag}>
          <Link href={`#`}>{mainCategory}</Link>
        </span>
        <span className={classes.duration}>{duration} mins to read</span>
        <span className={classes.pdfLink}>
          <Link href={`${pdfLink}`}>
            <a title={`Download ${id}.pdf`}>
              <img
                className={classes.pdf}
                width="16"
                src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAAECklEQVR4nO2aT4hNURzHPw+b8WcUQ2rIgpJGyjBIFkT5M0qpWYgiLFBKLCYbGxaklA2JIguGYhKzsBGyoBiGqPGv5E+YIf/HMDPX4swdx53z7p9z7jn3Pd6nTt175p3f73u/8zvn3nfugxIlSpQoUSIunsP2FmgE5jm5spi4NMBvPcAeIOfg+iLJwgC/7XRwfZFkaUAPsNL+JYYTFGWLMUANcAD4IeX7Dsy0mDcSVwbIrAvkfA2Mc5S7H1kYMFyRtxkY4ij/X2RhwAxFXg9oAgY60tBHFgbsU+T12x5HGvpwbcAY4LMir9zWOtDRh2sDzipyBttPYL4DLaBIbpu5wAXEM0CYCe3ARAd6MlkDQCyEjYQb8QAYaltIVgb4TAFOAt0KLR5wxLaArA3wmQZcUejpBibbTJyGAZuBR0AH0AIs04yTA3YoNO3XjBcLUwNWK2J0AbMNNB0NxLtrECsSUwPuKGJ4wCkDTYsDsT4axIrExIAc4tuch1jNa6Q47ww0TTfRNcAgcVL8cgdhxlPpfCQwSDNuV/RH8uPSAIDn0nEVYiH00d3y6tGX496Aq9JxHX8eXNqBX5oxnd6OTRfBWmnsJ+nYZOWuMtHlugIuIXZ0AMql/ocGMY0qwLUBXcAxRf8Ng5hGa0BS0ngSHAV8CcSZYaBpkoku1xUA0AYclM494JlBvKKaAj73peMcsNsgVtFNAYCLgTg9wAJjdQ5Iw4AKxDZWMNZboDIFjVZJw4Ct0vhm4EPg/DhwE3G77EB8f3gOnACqI2KXKTSWaepUYmpADmiVxl8Gniji5mudwJKQ+JWKMalWlakBSxUxkrawp8apis9PDRPk6i4wBzgDnM/zdw94HOh7B6wBxgObpP4JIXlGxOzTJmkFVAPXFOM8xPxuAjYCY3s/v53oLfDrIflWKD6/Iu7FxSGuAUOAQ+TfvW0g/8vNRcCLPOM6gVkheTcoxmyIvqz4xDFgBPm3vvz/fNTr7XJgG3ALcRdoA84hdn/CqFfkq48Yk4g4BpyR/t7Ve/5V6rP5QnOvQuPesAE2FsFa6XgVMIw/5d4K7LKQ0yfxIqi7DxfGe2Bw73GD1P8TYcg3Czl9EhtgowIOK/q6gfXAbQv5ZEbG7NMmzhowAPHGphWxz9cCLExTRAj36K/xXpoJTJ8EbfOK/hpfppmgEAyQ7zKm7XTS5IVgQAXwRqElaWsDRidNXggGACxXaEna6nQSF4oBYDYVEpe+TyEZoDsVtErfJ63Fx6TNkfToTAWt0i8kA64FNCWZCtqlX0gGeIgfRfjEnQpGpV9oBrTw92N8nKlgVPrFQNhUMC79YiDfVEil9IsF1VT450s/iDwV/ovSD+JPhVilb2NHKGvagS29x5E/v/sNJj1nsRXLWoQAAAAASUVORK5CYII="
              />
            </a>
          </Link>
        </span>
      </div>
    </div>
  );

  // return (
  //   <Box className={classes.tutorialCardWrapper}>
  //     <Typography component="p" className={classes.cardHeader}>
  //       <Link
  //         href={`/${url}`}
  //         target="_self"
  //
  //         className={classes.title}
  //       >
  //         {title}
  //       </Link>
  //     </Typography>
  //     <Stack
  //       direction="row"
  //       justifyContent="space-between"
  //       card-header={classes.cardDuration}
  //     >
  //       <Typography
  //         component="span"
  //         variant="body1"
  //         classes={{ root: classes.date }}
  //       >
  //         {duration} min
  //       </Typography>
  //       <Typography
  //         component="span"
  //         variant="body1"
  //         classes={{ root: classes.date }}
  //       >
  //         Updated {format(new Date(updated), 'LLL dd, yyyy')}
  //       </Typography>
  //     </Stack>
  //     <Stack
  //       direction="row"
  //       justifyContent="space-between"
  //       alignItems="center"
  //       className={classes.cardFooter}
  //     >
  //       <Link
  //         href={`/${url}`}
  //         target="_self"
  //
  //         className={classes.linkButton}
  //       >
  //         Start
  //       </Link>
  //       <Link href={`${pdfLink}`} title={`Download ${id}.pdf`}>
  //         <img
  //           className={classes.pdf}
  //           src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAAECklEQVR4nO2aT4hNURzHPw+b8WcUQ2rIgpJGyjBIFkT5M0qpWYgiLFBKLCYbGxaklA2JIguGYhKzsBGyoBiGqPGv5E+YIf/HMDPX4swdx53z7p9z7jn3Pd6nTt175p3f73u/8zvn3nfugxIlSpQoUSIunsP2FmgE5jm5spi4NMBvPcAeIOfg+iLJwgC/7XRwfZFkaUAPsNL+JYYTFGWLMUANcAD4IeX7Dsy0mDcSVwbIrAvkfA2Mc5S7H1kYMFyRtxkY4ij/X2RhwAxFXg9oAgY60tBHFgbsU+T12x5HGvpwbcAY4LMir9zWOtDRh2sDzipyBttPYL4DLaBIbpu5wAXEM0CYCe3ARAd6MlkDQCyEjYQb8QAYaltIVgb4TAFOAt0KLR5wxLaArA3wmQZcUejpBibbTJyGAZuBR0AH0AIs04yTA3YoNO3XjBcLUwNWK2J0AbMNNB0NxLtrECsSUwPuKGJ4wCkDTYsDsT4axIrExIAc4tuch1jNa6Q47ww0TTfRNcAgcVL8cgdhxlPpfCQwSDNuV/RH8uPSAIDn0nEVYiH00d3y6tGX496Aq9JxHX8eXNqBX5oxnd6OTRfBWmnsJ+nYZOWuMtHlugIuIXZ0AMql/ocGMY0qwLUBXcAxRf8Ng5hGa0BS0ngSHAV8CcSZYaBpkoku1xUA0AYclM494JlBvKKaAj73peMcsNsgVtFNAYCLgTg9wAJjdQ5Iw4AKxDZWMNZboDIFjVZJw4Ct0vhm4EPg/DhwE3G77EB8f3gOnACqI2KXKTSWaepUYmpADmiVxl8Gniji5mudwJKQ+JWKMalWlakBSxUxkrawp8apis9PDRPk6i4wBzgDnM/zdw94HOh7B6wBxgObpP4JIXlGxOzTJmkFVAPXFOM8xPxuAjYCY3s/v53oLfDrIflWKD6/Iu7FxSGuAUOAQ+TfvW0g/8vNRcCLPOM6gVkheTcoxmyIvqz4xDFgBPm3vvz/fNTr7XJgG3ALcRdoA84hdn/CqFfkq48Yk4g4BpyR/t7Ve/5V6rP5QnOvQuPesAE2FsFa6XgVMIw/5d4K7LKQ0yfxIqi7DxfGe2Bw73GD1P8TYcg3Czl9EhtgowIOK/q6gfXAbQv5ZEbG7NMmzhowAPHGphWxz9cCLExTRAj36K/xXpoJTJ8EbfOK/hpfppmgEAyQ7zKm7XTS5IVgQAXwRqElaWsDRidNXggGACxXaEna6nQSF4oBYDYVEpe+TyEZoDsVtErfJ63Fx6TNkfToTAWt0i8kA64FNCWZCtqlX0gGeIgfRfjEnQpGpV9oBrTw92N8nKlgVPrFQNhUMC79YiDfVEil9IsF1VT450s/iDwV/ovSD+JPhVilb2NHKGvagS29x5E/v/sNJj1nsRXLWoQAAAAASUVORK5CYII="
  //         />
  //       </Link>
  //     </Stack>
  //   </Box>
  // );
}
