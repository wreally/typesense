import chalk from "chalk";

export const LogLevel = {
  ERROR: 0,
  WARN: 1,
  INFO: 2,
  DEBUG: 4,
} as const;

export type LogLevelType = (typeof LogLevel)[keyof typeof LogLevel];
export type LogLevelName = keyof typeof LogLevel;

export interface Logger {
  error: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  success: (...args: unknown[]) => void;
  debug: (...args: unknown[]) => void;
  break: () => void;
  setLevel: (level: LogLevelType) => void;
  getLevel: () => LogLevelType;
  isLevelEnabled: (level: LogLevelType) => boolean;
}

type ChalkColor = (text: string) => string;

class LoggerSingleton implements Logger {
  private static instance: LoggerSingleton | null = null;
  private currentLevel: LogLevelType;
  private readonly chalkColors: {
    error: ChalkColor;
    warn: ChalkColor;
    info: ChalkColor;
    success: ChalkColor;
    debug: ChalkColor;
  };

  private constructor(initialLevel: LogLevelType = LogLevel.INFO) {
    this.currentLevel = initialLevel;
    this.chalkColors = {
      error: (text: string) => chalk.red(text),
      warn: (text: string) => chalk.yellow(text),
      info: (text: string) => chalk.white(text),
      success: (text: string) => chalk.green(text),
      debug: (text: string) => chalk.gray(text),
    };
  }

  public static getInstance(initialLevel: LogLevelType = LogLevel.INFO): LoggerSingleton {
    if (!LoggerSingleton.instance) {
      LoggerSingleton.instance = new LoggerSingleton(initialLevel);
    }
    return LoggerSingleton.instance;
  }

  public setLevel(level: LogLevelType): void {
    this.currentLevel = level;
  }

  public getLevel(): LogLevelType {
    return this.currentLevel;
  }

  public isLevelEnabled(level: LogLevelType): boolean {
    return this.currentLevel >= level;
  }

  private formatArg(arg: unknown): string {
    if (arg instanceof Error) {
      return arg.stack ?? arg.message;
    }
    return typeof arg === "string" ? arg : String(arg);
  }

  private formatMessage(args: unknown[]): string {
    return args.map((arg) => this.formatArg(arg)).join(" ");
  }

  private colorize(text: string, colorFn: ChalkColor): string {
    return colorFn(text);
  }

  public error(...args: unknown[]): void {
    if (this.isLevelEnabled(LogLevel.ERROR)) {
      const message = this.formatMessage(args);
      console.error(this.colorize(`\n${message}`, this.chalkColors.error));
    }
  }

  public warn(...args: unknown[]): void {
    if (this.isLevelEnabled(LogLevel.WARN)) {
      const message = this.formatMessage(args);
      console.warn(this.colorize(`\n${message}`, this.chalkColors.warn));
    }
  }

  public info(...args: unknown[]): void {
    if (this.isLevelEnabled(LogLevel.INFO)) {
      const message = this.formatMessage(args);
      console.info(this.colorize(`\n${message}`, this.chalkColors.info));
    }
  }

  public success(...args: unknown[]): void {
    if (this.isLevelEnabled(LogLevel.INFO)) {
      const message = this.formatMessage(args);
      console.log(this.colorize(`\n${message}`, this.chalkColors.success));
    }
  }

  public debug(...args: unknown[]): void {
    if (this.isLevelEnabled(LogLevel.DEBUG)) {
      const message = this.formatMessage(args);
      console.debug(this.colorize(`\n${message}`, this.chalkColors.debug));
    }
  }

  public break(): void {
    if (this.isLevelEnabled(LogLevel.INFO)) {
      console.log("");
    }
  }
}

export const logger = LoggerSingleton.getInstance();

export const LoggerClass = LoggerSingleton;
