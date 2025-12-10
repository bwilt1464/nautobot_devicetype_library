################################################################################
# Job to import desired device types into Nautobot from a YAML file.
#
# Created by: Bart Smeding
# Date: 2025-01-27
#
################################################################################

from nautobot.core.jobs import Job, StringVar, ChoiceVar, BooleanVar, register_jobs
import os
import yaml
import re
import shutil
from nautobot.dcim.models import (
    Manufacturer, DeviceType, InterfaceTemplate, ConsolePortTemplate, ConsoleServerPortTemplate,
    PowerPortTemplate, PowerOutletTemplate, FrontPortTemplate, RearPortTemplate,
    DeviceBay
)
from nautobot.extras.models import ImageAttachment
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.conf import settings

# Set the relative path to the device types folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate up from jobs/
DEVICE_TYPE_PATH = os.path.join(BASE_DIR, "device-types")
ELEVATION_IMAGE_PATH = os.path.join(BASE_DIR, "elevation-images")

name = "Import Device or Module Types"
class SyncDeviceTypes(Job):
    text_filter = StringVar(
        description="Enter text to filter device types (regex supported)",
        required=False
    )
    manufacturer = ChoiceVar(
        choices=[], 
        description="Select a manufacturer to import all device types",
        default="",
        required=False
    )
    dry_run = BooleanVar(
        description="Enable dry-run mode (only list files to be processed, no changes will be made)",
        default=True
    )
    debug_mode = BooleanVar(
        description="Enable debug mode for detailed logging",
        default=False
    )
    include_images = BooleanVar(
        description="Also import elevation images (front/rear) if available",
        default=False,
        required=False,
    )

    class Meta:
        name = "Sync Device Types"
        description = "Import device types from the local Nautobot Git repository with dry-run and debug options."
        job_class_name = "SyncDeviceTypes"
        soft_time_limit = 900  # 15 minutes in seconds
        time_limit = 960  # 16 minutes in seconds

    def run(self, *args, **kwargs):
        """Execute the job with dynamic argument handling."""
        debug_mode = kwargs.get("debug_mode", False)
        manufacturer = kwargs.get("manufacturer")
        text_filter = kwargs.get("text_filter")
        dry_run = kwargs.get("dry_run")
        include_images = kwargs.get("include_images", False)

        self.logger.info("Starting device type synchronization...")

        # Verify if the directory exists
        if not os.path.exists(DEVICE_TYPE_PATH):
            self.logger.error("Device types directory not found.")
            return


        if text_filter == '' and manufacturer == '':
            self.logger.error("No filter set, to prohibit excidentially import all device_types this is stopped.")
            return

        # Walk through the directory structure and log all folders and files
        if debug_mode:
            self.logger.info(f"Scanning directory structure under {DEVICE_TYPE_PATH}...")

        files_to_import = []

        for root, dirs, files in os.walk(DEVICE_TYPE_PATH):
            if debug_mode:
                self.logger.info(f"Checking folder: {root}")
            for file in files:
                if file.endswith(".yaml"):
                    full_path = os.path.join(root, file)
                    if manufacturer and manufacturer not in root:
                        continue  # Skip files not belonging to the selected manufacturer
                    if text_filter:
                        regex_pattern = re.compile(text_filter)
                        if not regex_pattern.search(file):
                            continue  # Skip files that do not match the regex filter
                    files_to_import.append(full_path)
                    self.logger.info(f"  Found file: {full_path}")

        if not files_to_import:
            self.logger.warning("No matching device type files found.")
            return

        # If dry-run is enabled, only list the files and, if requested, show potential images
        if dry_run:
            self.logger.info("Dry-run mode enabled. The following files would be processed:")
            for file_path in files_to_import:
                self.logger.info(f" - {file_path}")
                if include_images:
                    try:
                        with open(file_path, "r") as f:
                            dd = yaml.safe_load(f)
                        images_dir = self._resolve_manufacturer_images_dir(dd["manufacturer"])
                        if not images_dir:
                            self.logger.info("[Dry-run] No elevation-images directory for manufacturer")
                            continue
                        # Try using part_number first, then fall back to model
                        model_for_images = dd.get("part_number", dd["model"])
                        front_path, rear_path = self._find_elevation_image_paths(images_dir, dd["manufacturer"], model_for_images)
                        if front_path:
                            self.logger.info(f"[Dry-run] Would attach FRONT image: {front_path}")
                        if rear_path:
                            self.logger.info(f"[Dry-run] Would attach REAR image: {rear_path}")
                        if not front_path and not rear_path:
                            self.logger.info("[Dry-run] No elevation images found")
                    except Exception as img_err:
                        self.logger.warning(f"[Dry-run] Unable to evaluate images for {file_path}: {img_err}")
            self.logger.info("Dry-run completed successfully.")
            return

        # Process and import device types
        for file_path in files_to_import:
            try:
                with open(file_path, "r") as f:
                    device_data = yaml.safe_load(f)

                manufacturer_obj, _ = Manufacturer.objects.get_or_create(name=device_data["manufacturer"])
                device_type, created = DeviceType.objects.update_or_create(
                    model=device_data["model"],
                    manufacturer=manufacturer_obj,
                    defaults={
                        "part_number": device_data.get("part_number", ""),
                        "u_height": device_data.get("u_height", 1),
                        "is_full_depth": device_data.get("is_full_depth", True),
                        "comments": device_data.get("comments", ""),
                        "subdevice_role": device_data.get("subdevice_role", "") or "",
                    }
                )
                
                # Log device type creation/update
                self.logger.info(f"DeviceType {'created' if created else 'updated'}: {device_type.id} - {device_type.manufacturer.name} {device_type.model}")
                
                # Check if device type already has images (for debugging)
                if not created and include_images:
                    if hasattr(device_type, "front_image") and device_type.front_image:
                        self.logger.info(f"Existing front image: {device_type.front_image.name}")
                    if hasattr(device_type, "rear_image") and device_type.rear_image:
                        self.logger.info(f"Existing rear image: {device_type.rear_image.name}")

                if include_images:
                    try:
                        # Try using part_number first, then fall back to model
                        model_for_images = device_data.get("part_number", device_data["model"])
                        # Wrap image attachment in a transaction to ensure it's committed
                        with transaction.atomic():
                            self._attach_elevation_images(device_type, device_data["manufacturer"], model_for_images, commit=True, debug_mode=debug_mode)
                    except Exception as img_err:
                        self.logger.warning(f"Images not attached for {device_data['manufacturer']} {device_data['model']}: {img_err}")

                # Helper function to process components safely
                def process_component(component_list, component_model, fields, fk_field="device_type", parent_field="device_type_id", parent_value=None, defaults=None):
                    """Generic function to process different device components"""
                    if defaults is None:
                        defaults = {}
                    filter_kwargs = {fk_field: device_type}
                    component_model.objects.filter(**filter_kwargs).delete()
                    for item in device_data.get(component_list, []):
                        valid_data = {field: item.get(field, None) for field in fields if field in item}
                        # Apply defaults for missing fields
                        for field, default_value in defaults.items():
                            if field not in valid_data or valid_data[field] is None:
                                valid_data[field] = default_value
                        if parent_value:
                            valid_data[parent_field] = parent_value
                        valid_data[fk_field] = device_type
                        # Filter out any fields that are not in the fields list or defaults to avoid keyword argument errors
                        allowed_fields = fields + [fk_field] + list(defaults.keys())
                        filtered_data = {k: v for k, v in valid_data.items() if k in allowed_fields}
                        
                        # Create the component object
                        component_model.objects.create(**filtered_data)
                    self.logger.info(f"Checked {component_list} for {device_data['model']}.")


                # Define valid fields for each model with the correct foreign key
                process_component("interfaces", InterfaceTemplate, ["name", "type", "label", "description", "mgmt_only"])
                process_component("console-ports", ConsolePortTemplate, ["name", "type", "label", "description"])
                process_component("console-server-ports", ConsoleServerPortTemplate, ["name", "type", "label", "description"])
                # PowerPortTemplate disabled due to power_factor constraint issues
                # process_component("power-ports", PowerPortTemplate, ["name", "type", "maximum_draw", "allocated_draw", "power_factor"], defaults={"power_factor": 1.0})
                # Removed "power_port" from "power-outlets" due to "power-ports" template being disabled
                process_component("power-outlets", PowerOutletTemplate, ["name", "type", "feed_leg", "label", "description"])
                process_component("front-ports", FrontPortTemplate, ["name", "type", "rear_port", "rear_port_position", "label", "description"])
                process_component("rear-ports", RearPortTemplate, ["name", "type", "positions", "label", "description"])
                # process_component("module-bays", ModuleBay, ["name", "position"], fk_field="name", parent_field="device_type_id", parent_value=device_type.id)
                process_component("device-bays", DeviceBay, ["name", "label", "description"], fk_field="name", parent_field="device_type", parent_value=device_type)


                self.logger.info(f"Imported device type: {device_data['model']} (Created: {created})")
                
                # Final verification: check if images were attached
                if include_images:
                    try:
                        # Check DeviceType image fields
                        front_image_info = "None"
                        rear_image_info = "None"
                        
                        if hasattr(device_type, "front_image") and device_type.front_image:
                            front_image_info = f"{device_type.front_image.name} (URL: {device_type.front_image.url})"
                        
                        if hasattr(device_type, "rear_image") and device_type.rear_image:
                            rear_image_info = f"{device_type.rear_image.name} (URL: {device_type.rear_image.url})"
                        
                        self.logger.info(f"Final verification for {device_data['model']}:")
                        self.logger.info(f"  - Front image: {front_image_info}")
                        self.logger.info(f"  - Rear image: {rear_image_info}")
                        
                        # Also check ImageAttachments as fallback
                        ct = ContentType.objects.get_for_model(device_type)
                        attachments = ImageAttachment.objects.filter(
                            content_type=ct,
                            object_id=device_type.id
                        )
                        if attachments.exists():
                            self.logger.info(f"  - ImageAttachments: {attachments.count()} found")
                            for attachment in attachments:
                                self.logger.info(f"    - {attachment.name} (ID: {attachment.id})")
                    except Exception as verify_err:
                        self.logger.warning(f"Final verification failed: {verify_err}")
                        
            except Exception as e:
                self.logger.error(f"Failed to import {file_path}: {str(e)}")

        self.logger.info("Device type import completed successfully.")

    @classmethod
    def get_manufacturer_choices(cls):
        """Populate the manufacturer dropdown choices dynamically with an empty default value."""
        manufacturers = [("", "Select a manufacturer")]
        if os.path.exists(DEVICE_TYPE_PATH):
            for folder in os.listdir(DEVICE_TYPE_PATH):
                folder_path = os.path.join(DEVICE_TYPE_PATH, folder)
                if os.path.isdir(folder_path):
                    manufacturers.append((folder, folder))
        if len(manufacturers) == 1:
            manufacturers.append(("", "No manufacturers found"))

        return manufacturers

    @classmethod
    def as_form(cls, *args, **kwargs):
        manufacturers = cls.get_manufacturer_choices()
        if manufacturers:
            cls.manufacturer.choices = sorted(manufacturers, key=lambda x: x[1].lower())
        else:
            cls.manufacturer.choices = [("", "No manufacturers found")]
        form = super().as_form(*args, **kwargs)
        form.fields["manufacturer"].choices = cls.manufacturer.choices
        return form

    
    # ------------------------------
    # Image handling helpers
    # ------------------------------
    def _copy_image_to_media(self, source_path, target_filename, debug_mode=False):
        """Copy an image file directly to the media/devicetype-images/ directory.
        
        Returns the relative path to the copied file, or None if copy failed.
        """
        try:
            # Get the media root directory
            media_root = getattr(settings, 'MEDIA_ROOT', '/opt/nautobot/media')
            target_dir = os.path.join(media_root, 'devicetype-images')
            
            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Create the full target path
            target_path = os.path.join(target_dir, target_filename)
            
            # Check if target file already exists and has the same size
            if os.path.exists(target_path):
                try:
                    source_size = os.path.getsize(source_path)
                    target_size = os.path.getsize(target_path)
                    if source_size == target_size:
                        if debug_mode:
                            self.logger.debug(f"File already exists with same size, skipping copy: {target_filename}")
                        return f"devicetype-images/{target_filename}"
                    else:
                        if debug_mode:
                            self.logger.debug(f"File exists but size differs: {target_filename} (source={source_size}, target={target_size})")
                except OSError as size_err:
                    if debug_mode:
                        self.logger.debug(f"Could not check file sizes: {size_err}")
                    # Continue with copy attempt
            
            # Copy the file
            try:
                shutil.copy2(source_path, target_path)
            except PermissionError as perm_err:
                self.logger.error(f"Permission denied copying {target_filename}: {perm_err}")
                # Check if the file exists despite the error
                if os.path.exists(target_path):
                    self.logger.info(f"File {target_filename} exists despite permission error, using existing file")
                    return f"devicetype-images/{target_filename}"
                return None
            except OSError as os_err:
                self.logger.error(f"OS error copying {target_filename}: {os_err}")
                # Check if the file exists despite the error
                if os.path.exists(target_path):
                    self.logger.info(f"File {target_filename} exists despite OS error, using existing file")
                    return f"devicetype-images/{target_filename}"
                return None
            
            # Verify the copy was successful
            if os.path.exists(target_path):
                copied_size = os.path.getsize(target_path)
                source_size = os.path.getsize(source_path)
                if copied_size == source_size:
                    self.logger.info(f"Successfully copied image: {target_filename} ({copied_size} bytes)")
                    return f"devicetype-images/{target_filename}"
                else:
                    self.logger.error(f"File copy size mismatch: source={source_size}, copied={copied_size}")
                    return None
            else:
                self.logger.error(f"File copy failed: {target_path} does not exist")
                return None
                
        except Exception as copy_err:
            self.logger.error(f"Failed to copy image file {target_filename}: {copy_err}")
            # Check if the file actually exists despite the error
            if os.path.exists(target_path):
                self.logger.info(f"File {target_filename} exists despite error, using existing file")
                return f"devicetype-images/{target_filename}"
            return None
    def _attach_elevation_images(self, device_type, manufacturer_name, model_name, commit, debug_mode=False):
        """Attach front/rear elevation images to the given DeviceType if present on disk.

        Looks under elevation-images/<Manufacturer>/ for files named like
        "<manufacturer>-<model>.front.(png|jpg|jpeg)" (case-insensitive), and similar for rear.
        Falls back to using only the model in the filename if needed.
        """
        images_dir = self._resolve_manufacturer_images_dir(manufacturer_name)
        if not images_dir:
            self.logger.info(f"No elevation-images directory for manufacturer '{manufacturer_name}'. Skipping images.")
            return

        front_path, rear_path = self._find_elevation_image_paths(images_dir, manufacturer_name, model_name, debug_mode)

        if not front_path and not rear_path:
            self.logger.info(f"No elevation images found for {manufacturer_name} {model_name}.")
            return

        if not commit:
            if front_path:
                self.logger.info(f"[Dry-run] Would attach FRONT image: {front_path}")
            if rear_path:
                self.logger.info(f"[Dry-run] Would attach REAR image: {rear_path}")
            return

        # Attach FRONT: copy file directly to media folder and set DeviceType.front_image field
        if front_path:
            try:
                # Copy the file directly to the media folder
                filename = os.path.basename(front_path)
                relative_path = self._copy_image_to_media(front_path, filename, debug_mode)
                
                if relative_path:
                    # Set the DeviceType field to point to the copied file
                    device_type.front_image = relative_path
                    device_type.save()
                    self.logger.info(f"Successfully attached front image: {filename}")
                else:
                    self.logger.warning(f"Failed to copy front image, falling back to ImageAttachment")
                    self._attach_with_imageattachment(device_type, front_path, name_suffix="front elevation")
                    
            except Exception as img_err:
                self.logger.warning(f"Failed to set DeviceType.front_image: {img_err}")
                self._attach_with_imageattachment(device_type, front_path, name_suffix="front elevation")
            
            # Refresh the device type to ensure it's up to date
            device_type.refresh_from_db()

        # Attach REAR: copy file directly to media folder and set DeviceType.rear_image field
        if rear_path:
            try:
                # Copy the file directly to the media folder
                filename = os.path.basename(rear_path)
                relative_path = self._copy_image_to_media(rear_path, filename, debug_mode)
                
                if relative_path:
                    # Set the DeviceType field to point to the copied file
                    device_type.rear_image = relative_path
                    device_type.save()
                    self.logger.info(f"Successfully attached rear image: {filename}")
                else:
                    self.logger.warning(f"Failed to copy rear image, falling back to ImageAttachment")
                    self._attach_with_imageattachment(device_type, rear_path, name_suffix="rear elevation")
                    
            except Exception as img_err:
                self.logger.warning(f"Failed to set DeviceType.rear_image: {img_err}")
                self._attach_with_imageattachment(device_type, rear_path, name_suffix="rear elevation")
            
            # Final refresh and verification
            device_type.refresh_from_db()
            
            # Log summary of attached images
            attached_images = []
            if device_type.front_image:
                attached_images.append("front")
            if device_type.rear_image:
                attached_images.append("rear")
            
            if attached_images:
                self.logger.info(f"Successfully attached {', '.join(attached_images)} image(s) to {manufacturer_name} {model_name}")
            else:
                self.logger.warning(f"No images were attached to {manufacturer_name} {model_name}")

    def _attach_with_imageattachment(self, device_type, image_path, name_suffix):
        """Create or replace an ImageAttachment for the given object."""
        ct = ContentType.objects.get_for_model(device_type)
        # Remove existing similar-named attachments to avoid duplicates
        try:
            ImageAttachment.objects.filter(content_type=ct, object_id=device_type.id, name__icontains=name_suffix).delete()
        except Exception:
            pass
        
        # Verify the source image file exists and is readable
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Source image file not found: {image_path}")
        
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"Cannot read source image file: {image_path}")
        
        try:
            # Create a Django File object directly from the file path
            django_file = File(open(image_path, "rb"), name=os.path.basename(image_path))
            
            # Use a transaction to ensure the attachment is properly saved
            with transaction.atomic():
                img = ImageAttachment.objects.create(
                    content_type=ct,
                    object_id=device_type.id,
                    name=f"{device_type.manufacturer.name} {device_type.model} {name_suffix}",
                    image=django_file,
                )
            
            # Close the file handle
            django_file.close()
                
            # Verify the attachment was created and saved
            saved_path = img.image.path
            exists = os.path.exists(saved_path)
            url = getattr(img.image, "url", None)
            file_size = os.path.getsize(saved_path) if exists else 0
            
            self.logger.info(f"Attachment created successfully:")
            self.logger.info(f"  - ID: {img.id}")
            self.logger.info(f"  - Name: {img.name}")
            self.logger.info(f"  - Content Type: {img.content_type}")
            self.logger.info(f"  - Object ID: {img.object_id}")
            self.logger.info(f"  - Device Type ID: {device_type.id}")
            self.logger.info(f"  - Stored at: {saved_path}")
            self.logger.info(f"  - File exists: {exists}")
            self.logger.info(f"  - File size: {file_size} bytes")
            self.logger.info(f"  - URL: {url}")
            
            # Verify the attachment is properly linked to the device type
            if img.content_type != ct:
                self.logger.warning(f"Content type mismatch: expected {ct}, got {img.content_type}")
            if img.object_id != device_type.id:
                self.logger.warning(f"Object ID mismatch: expected {device_type.id}, got {img.object_id}")
            
            if not exists:
                self.logger.warning(f"Image file was not saved to expected location: {saved_path}")
            
            # Test if we can query the attachment back
            try:
                test_query = ImageAttachment.objects.filter(
                    content_type=ct,
                    object_id=device_type.id,
                    name__icontains=name_suffix
                )
                self.logger.info(f"  - Query test: Found {test_query.count()} matching attachments")
                if test_query.exists():
                    self.logger.info(f"  - Query test: First attachment ID: {test_query.first().id}")
            except Exception as query_err:
                self.logger.warning(f"Query test failed: {query_err}")
                
        except Exception as img_err:
            self.logger.error(f"Failed to create ImageAttachment: {img_err}")
            self.logger.error(f"Source file: {image_path}")
            self.logger.error(f"Device type: {device_type.manufacturer.name} {device_type.model}")
            raise

    def _find_elevation_image_paths(self, images_dir, manufacturer_name, model_name, debug_mode=False):
        """Return (front_path, rear_path) for the given manufacturer/model if found.

        Matches filenames in a case-insensitive way by normalizing to lowercase.
        Candidate stems tried:
          - <slug(manufacturer)>-<slug(model)>
          - <slug(model)>
        """
        if not os.path.isdir(images_dir):
            return (None, None)

        # Build a lookup of lowercase filename -> real path
        filename_to_path = {}
        for root, _, files in os.walk(images_dir):
            for fname in files:
                filename_to_path[fname.lower()] = os.path.join(root, fname)

        manufacturer_slug = self._slugify(manufacturer_name)
        model_slug = self._slugify(model_name)
        
        # Debug logging
        if debug_mode and hasattr(self, 'logger'):
            self.logger.debug(f"Looking for images: manufacturer='{manufacturer_name}' -> '{manufacturer_slug}', model='{model_name}' -> '{model_slug}'")
            self.logger.debug(f"Found {len(filename_to_path)} image files in directory")

        # Generate base variants: original model slug and versions with common prefixes removed
        base_variants = []
        if model_slug:
            base_variants.append(model_slug)
            # Remove common marketing prefixes like 'catalyst-'
            common_prefixes = ["catalyst-"]
            for prefix in common_prefixes:
                if model_slug.startswith(prefix):
                    base_variants.append(model_slug[len(prefix):])

        # Add variants where first token is prefixed with 'c' (e.g., 9300-48t -> c9300-48t)
        augmented_variants = list(base_variants)
        for variant in base_variants:
            parts = variant.split("-") if variant else []
            if parts and not parts[0].startswith("c"):
                prefixed = "-".join(["c" + parts[0]] + parts[1:])
                if prefixed not in augmented_variants:
                    augmented_variants.append(prefixed)
        base_variants = augmented_variants

        # Build progressive stems from each base variant by removing trailing segments
        progressive_stems = []
        for variant in base_variants:
            current = variant
            while True:
                if current and current not in progressive_stems:
                    progressive_stems.append(current)
                if not current or "-" not in current:
                    break
                current = current.rsplit("-", 1)[0]

        # Combine with manufacturer-prefixed variants
        candidate_stems = []
        for stem in progressive_stems:
            candidate_stems.append(stem)
            candidate_stems.append(f"{manufacturer_slug}-{stem}")
        extensions = ["png", "jpg", "jpeg"]

        front_path = None
        rear_path = None
        
        # Debug logging for candidate stems
        if debug_mode and hasattr(self, 'logger'):
            self.logger.debug(f"Trying {len(candidate_stems)} candidate stems")
        
        for stem in candidate_stems:
            if not front_path:
                for ext in extensions:
                    key = f"{stem}.front.{ext}"
                    if key in filename_to_path:
                        front_path = filename_to_path[key]
                        break
            if not rear_path:
                for ext in extensions:
                    key = f"{stem}.rear.{ext}"
                    if key in filename_to_path:
                        rear_path = filename_to_path[key]
                        break
            if front_path and rear_path:
                break

        if debug_mode and hasattr(self, 'logger'):
            self.logger.debug(f"Final result: front={front_path}, rear={rear_path}")
        
        return (front_path, rear_path)

    def _resolve_manufacturer_images_dir(self, manufacturer_name):
        """Resolve the images directory for a manufacturer in a case/slug-insensitive way."""
        images_root = os.path.join(BASE_DIR, "elevation-images")
        if not os.path.isdir(images_root):
            return None
        target_lower = str(manufacturer_name).lower()
        target_slug = self._slugify(manufacturer_name)
        try:
            for entry in os.listdir(images_root):
                full_path = os.path.join(images_root, entry)
                if not os.path.isdir(full_path):
                    continue
                if entry.lower() == target_lower or self._slugify(entry) == target_slug:
                    return full_path
        except Exception:
            return None
        return None

    def _slugify(self, value):
        """Simplistic slugify to align with devicetype-library filenames."""
        value = str(value).strip().lower()
        # Replace whitespace and underscores with hyphens
        value = re.sub(r"[\s_]+", "-", value)
        # Remove any character that's not alphanumeric or hyphen
        value = re.sub(r"[^a-z0-9-]", "", value)
        # Collapse multiple hyphens
        value = re.sub(r"-+", "-", value)
        return value

# Register the job explicitly
register_jobs(SyncDeviceTypes)

