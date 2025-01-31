import type { Ora } from "ora";

import ora from "ora";

import { DockerService } from "@/services/docker";
import { FilesystemService } from "@/services/fs";
import { GitService } from "@/services/git";
import { TypesenseDirectoryManager } from "@/services/typesense-dir";

interface ServiceTypes {
  docker: DockerService;
  fs: FilesystemService;
  hostGit: GitService;
  containerGit: GitService;
  typesense: TypesenseDirectoryManager;
}

export class ServiceContainer {
  private readonly spinner: Ora;
  private readonly rootDir: string;
  private services: Partial<ServiceTypes> = {};

  constructor(rootDir: string) {
    this.spinner = ora();
    this.rootDir = rootDir;
  }

  initialize(options: {
    directory: string;
    gitUrl: string;
    containerName: string;
    typesenseDirectory?: string;
    yesToAll?: boolean;
    binaryPath?: string;
  }) {
    const fsService = new FilesystemService(this.spinner, options.yesToAll);
    const dockerService = new DockerService(this.spinner, fsService, this.rootDir);

    // Create Git service for host operations
    const hostGitService = new GitService(this.spinner, options.gitUrl);

    // Create Git service for container operations
    const containerGitService = new GitService(this.spinner, options.gitUrl, undefined, {
      containerName: options.containerName,
      dockerService: dockerService,
    });

    const typesenseManager = new TypesenseDirectoryManager(
      hostGitService, // Use host git service for TypesenseDirectoryManager
      fsService,
      dockerService,
      this.spinner,
      {
        typesenseGitUrl: options.gitUrl,
        yesToAll: options.yesToAll,
      },
    );

    this.services = {
      fs: fsService,
      docker: dockerService,
      hostGit: hostGitService,
      containerGit: containerGitService,
      typesense: typesenseManager,
    };
  }

  get<T extends keyof ServiceTypes>(serviceName: T): ServiceTypes[T] {
    const service = this.services[serviceName];
    if (!service) {
      throw new Error(`${serviceName} service not initialized. Call initialize() first.`);
    }
    return service as ServiceTypes[T];
  }

  getSpinner(): Ora {
    return this.spinner;
  }
}
